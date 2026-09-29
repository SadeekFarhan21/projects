#include "vio/sim.hpp"

#include <Eigen/Cholesky>
#include <cmath>

namespace vio::sim {
namespace {

Mat3 AttitudeAt(double t) {
  const double yaw = 0.8 * std::sin(0.25 * t);
  const double pitch = 0.15 * std::sin(0.6 * t);
  const double roll = 0.12 * std::sin(0.9 * t + 0.3);
  return Exp(Vec3(0, 0, yaw)) * Exp(Vec3(0, pitch, 0)) * Exp(Vec3(roll, 0, 0));
}

double NormalQuantile(double p) {
  // Acklam-style rational approximation is overkill; bisection on erf is fine.
  double lo = -10, hi = 10;
  for (int i = 0; i < 100; ++i) {
    const double mid = 0.5 * (lo + hi);
    const double cdf = 0.5 * std::erfc(-mid / std::sqrt(2.0));
    (cdf < p ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace

TruthSample Lissajous(double t) {
  TruthSample s;
  s.t = t;
  const double A = 3.0, B = 2.0, C = 0.5;
  const double a = 0.4, b = 0.6, c = 0.8;
  s.p = Vec3(A * std::sin(a * t), B * std::sin(b * t), 1.0 + C * std::sin(c * t));
  s.v = Vec3(A * a * std::cos(a * t), B * b * std::cos(b * t), C * c * std::cos(c * t));
  s.a = Vec3(-A * a * a * std::sin(a * t), -B * b * b * std::sin(b * t), -C * c * c * std::sin(c * t));
  s.R = AttitudeAt(t);
  // Body rate by central difference on the group; error is O(h^2).
  const double h = 1e-4;
  s.w = Log(AttitudeAt(t - h).transpose() * AttitudeAt(t + h)) / (2 * h);
  return s;
}

ImuSample IdealImu(const TruthSample& s, const Vec3& g) {
  ImuSample m;
  m.t = s.t;
  m.gyro = s.w;
  m.acc = s.R.transpose() * (s.a - g);
  return m;
}

Mat6 MeasNoise::Cov() const {
  Mat6 C = Mat6::Zero();
  C.topLeftCorner<3, 3>() = rot_sigma * rot_sigma * Mat3::Identity();
  C.bottomRightCorner<3, 3>() = pos_sigma * pos_sigma * Mat3::Identity();
  return C;
}

Mat15 DefaultP0() {
  Eigen::Matrix<double, 15, 1> d;
  d << Vec3::Constant(1e-4), Vec3::Constant(1e-3), Vec3::Constant(std::pow(0.5 * M_PI / 180, 2)),
      Vec3::Constant(1e-6), Vec3::Constant(1e-3);
  return d.asDiagonal();
}

double Chi2Quantile(double p, double k) {
  const double z = NormalQuantile(p);
  const double h = 2.0 / (9.0 * k);
  return k * std::pow(1.0 - h + z * std::sqrt(h), 3);
}

std::vector<McStep> RunMonteCarlo(const McRunConfig& cfg) {
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<double> N(0.0, 1.0);
  auto randn3 = [&] { return Vec3(N(rng), N(rng), N(rng)); };

  // Sample the true initial error from P0 and build the filter's initial guess.
  const Eigen::Matrix<double, 15, 15> L = cfg.P0.llt().matrixL();
  Eigen::Matrix<double, 15, 1> z;
  for (int i = 0; i < 15; ++i) z(i) = N(rng);
  const Eigen::Matrix<double, 15, 1> e0 = L * z;

  const TruthSample s0 = Lissajous(0.0);
  Vec3 bg_true = Vec3(0.002, -0.001, 0.0015), ba_true = Vec3(0.03, -0.02, 0.05);
  NominalState x0;
  x0.t = 0.0;
  // truth = estimate (+) error  =>  estimate = truth (-) error
  x0.p = s0.p - e0.segment<3>(idx::P);
  x0.v = s0.v - e0.segment<3>(idx::V);
  x0.R = s0.R * Exp(-e0.segment<3>(idx::TH));
  x0.bg = bg_true - e0.segment<3>(idx::BG);
  x0.ba = ba_true - e0.segment<3>(idx::BA);

  ImuNoise fn = cfg.imu;
  fn.acc_noise *= cfg.filter_imu_scale;
  fn.gyro_noise *= cfg.filter_imu_scale;
  fn.acc_walk *= cfg.filter_imu_scale;
  fn.gyro_walk *= cfg.filter_imu_scale;
  Eskf f;
  f.Init(x0, cfg.P0, fn);

  const double dt = 1.0 / cfg.imu_rate;
  const int per_cam = static_cast<int>(std::round(cfg.imu_rate / cfg.cam_rate));
  const int steps = static_cast<int>(cfg.duration * cfg.imu_rate);
  const Mat6 Rm = cfg.meas.Cov();
  const Eigen::Matrix<double, 6, 6> Lm = Rm.llt().matrixL();

  auto noisy = [&](const TruthSample& s) {
    ImuSample m = IdealImu(s);
    m.gyro += bg_true + randn3() * cfg.imu.gyro_noise / std::sqrt(dt);
    m.acc += ba_true + randn3() * cfg.imu.acc_noise / std::sqrt(dt);
    return m;
  };

  std::vector<McStep> out;
  TruthSample prev_cam = s0;
  ImuSample prev = noisy(s0);
  for (int k = 1; k <= steps; ++k) {
    // Bias random walk over the interval.
    bg_true += randn3() * cfg.imu.gyro_walk * std::sqrt(dt);
    ba_true += randn3() * cfg.imu.acc_walk * std::sqrt(dt);
    const TruthSample s = Lissajous(k * dt);
    const ImuSample cur = noisy(s);
    f.Propagate(prev, cur);
    prev = cur;
    if (k % per_cam != 0) continue;

    // Relative pose measurement between previous and current camera time.
    const Mat3 R12 = prev_cam.R.transpose() * s.R;
    const Vec3 p12 = prev_cam.R.transpose() * (s.p - prev_cam.p);
    Vec6 n;
    for (int i = 0; i < 6; ++i) n(i) = N(rng);
    n = Lm * n;
    f.UpdateRelativePose(R12 * Exp(n.head<3>()), p12 + n.tail<3>(), Rm, -1.0);
    f.Clone();
    prev_cam = s;

    const NominalState& x = f.state();
    Eigen::Matrix<double, 15, 1> e;
    e.segment<3>(idx::P) = s.p - x.p;
    e.segment<3>(idx::V) = s.v - x.v;
    e.segment<3>(idx::TH) = Log(x.R.transpose() * s.R);
    e.segment<3>(idx::BG) = bg_true - x.bg;
    e.segment<3>(idx::BA) = ba_true - x.ba;
    const Mat15 P = f.cov().topLeftCorner<15, 15>();
    McStep st;
    st.t = s.t;
    st.nees_core = e.dot(P.ldlt().solve(e));
    Vec6 ep;
    ep << e.segment<3>(idx::P), e.segment<3>(idx::TH);
    Mat6 Pp;
    Pp << P.block<3, 3>(idx::P, idx::P), P.block<3, 3>(idx::P, idx::TH), P.block<3, 3>(idx::TH, idx::P),
        P.block<3, 3>(idx::TH, idx::TH);
    st.nees_pose = ep.dot(Pp.ldlt().solve(ep));
    st.pos_err = e.segment<3>(idx::P).norm();
    out.push_back(st);
  }
  return out;
}

}  // namespace vio::sim
