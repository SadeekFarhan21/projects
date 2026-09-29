// Minimal EuRoC ASL-format reader (mav0/ directory layout).
#pragma once

#include <string>
#include <vector>

#include "vio/geometry.hpp"
#include "vio/imu.hpp"

namespace vio {

struct StereoFrameInfo {
  double t;
  std::string left_path;
  std::string right_path;
};

struct GroundTruthSample {
  double t;
  Vec3 p;
  Quat q;  // R_WB
  Vec3 v, bg, ba;
};

struct EurocSequence {
  Camera cam0, cam1;
  ImuNoise imu_noise;
  std::vector<ImuSample> imu;
  std::vector<StereoFrameInfo> frames;
  std::vector<GroundTruthSample> gt;
};

// Throws std::runtime_error on missing files.
EurocSequence LoadEuroc(const std::string& mav0_dir);

// Parse a camera sensor.yaml (T_BS, intrinsics, distortion, resolution).
Camera ParseCameraYaml(const std::string& path);

}  // namespace vio
