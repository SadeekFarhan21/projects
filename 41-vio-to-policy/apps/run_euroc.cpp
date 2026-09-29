// Run the stereo VIO on a EuRoC sequence and write a TUM trajectory plus a
// per-frame log (tracking stats, NIS, timings).
//
//   run_euroc --data data/MH_01_easy/mav0 --out results/mh01 [--init static|gt]
//             [--no-vision] [--meas-scale S] [--imu-scale S] [--features N]
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "vio/euroc.hpp"
#include "vio/sim.hpp"
#include "vio/vio_system.hpp"

using namespace vio;

namespace {

std::map<std::string, std::string> ParseArgs(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) continue;
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      a[k.substr(2)] = argv[++i];
    } else {
      a[k.substr(2)] = "1";
    }
  }
  return a;
}

// First 1 s IMU window (after t_min) that looks stationary.
int FindStaticWindow(const std::vector<ImuSample>& imu, double t_min, double* t_end) {
  const int W = 200;
  for (size_t s = 0; s + W < imu.size(); s += 20) {
    if (imu[s].t < t_min) continue;
    Vec3 mean = Vec3::Zero();
    double wmax = 0;
    for (int k = 0; k < W; ++k) {
      mean += imu[s + k].acc;
      wmax = std::max(wmax, imu[s + k].gyro.norm());
    }
    mean /= W;
    double var = 0;
    for (int k = 0; k < W; ++k) var += (imu[s + k].acc - mean).squaredNorm();
    var /= W;
    if (std::sqrt(var) < 0.05 && wmax < 0.05) {
      *t_end = imu[s + W - 1].t;
      return static_cast<int>(s);
    }
  }
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  auto args = ParseArgs(argc, argv);
  const std::string data = args.count("data") ? args["data"] : "data/MH_01_easy/mav0";
  const std::string out = args.count("out") ? args["out"] : "results/mh01";
  const std::string init = args.count("init") ? args["init"] : "static";
  std::filesystem::create_directories(std::filesystem::path(out).parent_path());

  EurocSequence seq = LoadEuroc(data);
  std::cerr << "loaded " << seq.frames.size() << " stereo frames, " << seq.imu.size()
            << " imu samples, " << seq.gt.size() << " gt samples\n";

  VioOptions opt;
  if (args.count("no-vision")) opt.use_vision = false;
  if (args.count("meas-scale")) opt.meas_cov_scale = std::stod(args["meas-scale"]);
  if (args.count("imu-scale")) opt.imu_noise_scale = std::stod(args["imu-scale"]);
  if (args.count("features")) opt.frontend.max_features = std::stoi(args["features"]);
  if (args.count("gate")) opt.gate_chi2 = std::stod(args["gate"]);

  NominalState x0;
  Mat15 P0 = Mat15::Zero();
  if (init == "gt") {
    // Start at the first frame covered by ground truth.
    size_t f = 0;
    while (f < seq.frames.size() && seq.frames[f].t < seq.gt.front().t) ++f;
    const double t0 = seq.frames[f].t;
    size_t g = 0;
    while (g + 1 < seq.gt.size() && seq.gt[g].t < t0) ++g;
    const auto& G = seq.gt[g];
    x0.t = t0;
    x0.p = G.p;
    x0.v = G.v;
    x0.R = G.q.toRotationMatrix();
    x0.bg = G.bg;
    x0.ba = G.ba;
    P0 = sim::DefaultP0();
    P0.block<3, 3>(idx::BG, idx::BG) = 1e-6 * Mat3::Identity();
    P0.block<3, 3>(idx::BA, idx::BA) = 1e-4 * Mat3::Identity();
  } else {
    double t_end = 0;
    const int s = FindStaticWindow(seq.imu, seq.imu.front().t, &t_end);
    if (s < 0) {
      std::cerr << "no static window found\n";
      return 1;
    }
    std::vector<ImuSample> w(seq.imu.begin() + s, seq.imu.begin() + s + 200);
    x0 = StaticInit(w);
    std::cerr << "static init window ends at +" << (t_end - seq.imu.front().t) << " s\n";
    Eigen::Matrix<double, 15, 1> d;
    const double deg = M_PI / 180;
    d << Vec3::Constant(1e-6), Vec3::Constant(1e-4), Vec3(std::pow(1 * deg, 2), std::pow(1 * deg, 2), 1e-4),
        Vec3::Constant(1e-6), Vec3::Constant(0.1 * 0.1);
    P0 = d.asDiagonal();
  }

  VioSystem vio(seq.cam0, seq.cam1, seq.imu_noise, opt);
  vio.Initialize(x0, P0);

  FILE* ftraj = std::fopen((out + "_traj_tum.txt").c_str(), "w");
  FILE* flog = std::fopen((out + "_frames.csv").c_str(), "w");
  std::fprintf(flog,
               "t,n_tracked,n_inliers,n_features,n_stereo,rms_px,nis,accepted,track_ms,pnp_ms,"
               "detect_ms,stereo_ms,filter_ms,total_ms,load_ms,bgx,bgy,bgz,bax,bay,baz,sig_px,sig_py,"
               "sig_pz\n");

  size_t imu_i = 0;
  int n_frames = 0, n_acc = 0;
  double sum_total = 0;
  for (const auto& fr : seq.frames) {
    if (fr.t <= x0.t) continue;
    while (imu_i < seq.imu.size() && seq.imu[imu_i].t <= fr.t + 0.01) vio.AddImu(seq.imu[imu_i++]);
    auto tl0 = std::chrono::steady_clock::now();
    cv::Mat L = cv::imread(fr.left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat R = cv::imread(fr.right_path, cv::IMREAD_GRAYSCALE);
    auto tl1 = std::chrono::steady_clock::now();
    if (L.empty() || R.empty()) continue;
    const FrameLog lg = vio.AddStereo(fr.t, L, R);
    const auto& x = lg.x;
    const Quat q(x.R);
    std::fprintf(ftraj, "%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", fr.t, x.p.x(), x.p.y(), x.p.z(),
                 q.x(), q.y(), q.z(), q.w());
    const auto& P = vio.filter().cov();
    const double load_ms = std::chrono::duration<double, std::milli>(tl1 - tl0).count();
    std::fprintf(flog,
                 "%.9f,%d,%zu,%d,%d,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,"
                 "%.6f,%.6f,%.6f,%.6f,%.6f\n",
                 fr.t, lg.fe.n_tracked, lg.fe.pnp.inliers.size(), lg.fe.n_features, lg.fe.n_stereo,
                 lg.fe.pnp.rms_px, lg.upd.nis, lg.upd.accepted ? 1 : 0, lg.fe.timing.track_ms,
                 lg.fe.timing.pnp_ms, lg.fe.timing.detect_ms, lg.fe.timing.stereo_ms, lg.filter_ms,
                 lg.total_ms, load_ms, x.bg.x(), x.bg.y(), x.bg.z(), x.ba.x(), x.ba.y(), x.ba.z(),
                 std::sqrt(P(0, 0)), std::sqrt(P(1, 1)), std::sqrt(P(2, 2)));
    ++n_frames;
    n_acc += lg.upd.accepted;
    sum_total += lg.total_ms;
  }
  std::fclose(ftraj);
  std::fclose(flog);
  std::cerr << "frames " << n_frames << ", accepted updates " << n_acc << ", mean total ms "
            << sum_total / std::max(1, n_frames) << "\n";
  return 0;
}
