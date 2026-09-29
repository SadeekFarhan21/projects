#include "vio/euroc.hpp"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace vio {
namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Extract the numeric list after `key:` in a flat [a, b, c] form, spanning lines.
std::vector<double> ListAfter(const std::string& text, const std::string& key) {
  const auto k = text.find(key + ":");
  if (k == std::string::npos) throw std::runtime_error("missing key " + key);
  const auto lb = text.find('[', k);
  const auto rb = text.find(']', lb);
  std::string body = text.substr(lb + 1, rb - lb - 1);
  for (char& c : body)
    if (c == ',') c = ' ';
  std::stringstream ss(body);
  std::vector<double> out;
  double v;
  while (ss >> v) out.push_back(v);
  return out;
}

double ScalarAfter(const std::string& text, const std::string& key) {
  std::regex re(key + R"(:\s*([-+0-9.eE]+))");
  std::smatch m;
  if (!std::regex_search(text, m, re)) throw std::runtime_error("missing key " + key);
  return std::stod(m[1]);
}

std::vector<std::vector<double>> ReadCsv(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
    rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<std::pair<long long, std::string>> ReadImageList(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<std::pair<long long, std::string>> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto c = line.find(',');
    std::string name = line.substr(c + 1);
    while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();
    out.emplace_back(std::stoll(line.substr(0, c)), name);
  }
  return out;
}

}  // namespace

Camera ParseCameraYaml(const std::string& path) {
  const std::string text = ReadFile(path);
  Camera c;
  const auto T = ListAfter(text, "data");
  if (T.size() != 16) throw std::runtime_error("bad T_BS in " + path);
  Mat4 M;
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k) M(r, k) = T[4 * r + k];
  c.T_BC = Pose::FromMatrix(M);
  c.T_BC.R = Orthonormalize(c.T_BC.R);
  const auto K = ListAfter(text, "intrinsics");
  c.fx = K[0];
  c.fy = K[1];
  c.cx = K[2];
  c.cy = K[3];
  const auto D = ListAfter(text, "distortion_coefficients");
  c.k1 = D[0];
  c.k2 = D[1];
  c.p1 = D[2];
  c.p2 = D[3];
  const auto res = ListAfter(text, "resolution");
  c.width = static_cast<int>(res[0]);
  c.height = static_cast<int>(res[1]);
  return c;
}

EurocSequence LoadEuroc(const std::string& mav0) {
  EurocSequence s;
  s.cam0 = ParseCameraYaml(mav0 + "/cam0/sensor.yaml");
  s.cam1 = ParseCameraYaml(mav0 + "/cam1/sensor.yaml");
  const std::string imu_yaml = ReadFile(mav0 + "/imu0/sensor.yaml");
  s.imu_noise.gyro_noise = ScalarAfter(imu_yaml, "gyroscope_noise_density");
  s.imu_noise.gyro_walk = ScalarAfter(imu_yaml, "gyroscope_random_walk");
  s.imu_noise.acc_noise = ScalarAfter(imu_yaml, "accelerometer_noise_density");
  s.imu_noise.acc_walk = ScalarAfter(imu_yaml, "accelerometer_random_walk");

  for (const auto& r : ReadCsv(mav0 + "/imu0/data.csv")) {
    ImuSample m;
    m.t = r[0] * 1e-9;
    m.gyro = Vec3(r[1], r[2], r[3]);
    m.acc = Vec3(r[4], r[5], r[6]);
    s.imu.push_back(m);
  }
  const auto left = ReadImageList(mav0 + "/cam0/data.csv");
  const auto right = ReadImageList(mav0 + "/cam1/data.csv");
  // Pair frames by identical timestamps (EuRoC stereo is hardware-synced).
  size_t j = 0;
  for (const auto& [ts, name] : left) {
    while (j < right.size() && right[j].first < ts) ++j;
    if (j < right.size() && right[j].first == ts) {
      s.frames.push_back({ts * 1e-9, mav0 + "/cam0/data/" + name, mav0 + "/cam1/data/" + right[j].second});
    }
  }
  try {
    for (const auto& r : ReadCsv(mav0 + "/state_groundtruth_estimate0/data.csv")) {
      GroundTruthSample g;
      g.t = r[0] * 1e-9;
      g.p = Vec3(r[1], r[2], r[3]);
      g.q = Quat(r[4], r[5], r[6], r[7]).normalized();
      g.v = Vec3(r[8], r[9], r[10]);
      g.bg = Vec3(r[11], r[12], r[13]);
      g.ba = Vec3(r[14], r[15], r[16]);
      s.gt.push_back(g);
    }
  } catch (const std::runtime_error&) {
    // Ground truth is optional.
  }
  return s;
}

}  // namespace vio
