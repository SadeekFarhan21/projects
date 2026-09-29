// Glue between the IMU stream, the stereo frontend and the ESKF.
#pragma once

#include <memory>
#include <vector>

#include "vio/eskf.hpp"
#include "vio/frontend.hpp"

namespace vio {

struct VioOptions {
  FrontendOptions frontend;
  double gate_chi2 = 22.46;     // 6 dof, 99.9 percent
  double meas_cov_scale = 1.0;  // inflation of the PnP-derived covariance
  double imu_noise_scale = 1.0; // inflation of the datasheet IMU densities
  bool use_vision = true;       // false = IMU dead reckoning only
};

struct FrameLog {
  double t = 0;
  NominalState x;
  FrontendResult fe;
  UpdateResult upd;
  double filter_ms = 0;  // IMU propagation + update + clone
  double total_ms = 0;   // image load excluded, frontend + filter included
};

class VioSystem {
 public:
  VioSystem(const Camera& cam0, const Camera& cam1, const ImuNoise& noise, const VioOptions& opt);

  // Initialize at time t with a given state and core covariance.
  void Initialize(const NominalState& x0, const Mat15& P0);

  // Queue IMU samples (must be time ordered).
  void AddImu(const ImuSample& m);

  // Propagate to the frame time, run the frontend and fuse. Returns a log entry.
  FrameLog AddStereo(double t, const cv::Mat& left, const cv::Mat& right);

  const Eskf& filter() const { return eskf_; }

 private:
  void PropagateTo(double t);

  Camera cam0_;
  VioOptions opt_;
  ImuNoise noise_;
  Eskf eskf_;
  std::unique_ptr<StereoFrontend> fe_;
  std::vector<ImuSample> buf_;
  ImuSample last_;  // last sample consumed by the filter (possibly interpolated)
  bool initialized_ = false;
};

// Static initialization from a window of IMU samples at rest: roll and pitch
// from mean specific force, yaw zero, gyro bias from mean rate.
NominalState StaticInit(const std::vector<ImuSample>& window);

}  // namespace vio
