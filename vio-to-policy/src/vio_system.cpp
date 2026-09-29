#include "vio/vio_system.hpp"

#include <chrono>

namespace vio {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
ImuSample Lerp(const ImuSample& a, const ImuSample& b, double t) {
  const double s = (t - a.t) / (b.t - a.t);
  ImuSample m;
  m.t = t;
  m.gyro = (1 - s) * a.gyro + s * b.gyro;
  m.acc = (1 - s) * a.acc + s * b.acc;
  return m;
}
}  // namespace

VioSystem::VioSystem(const Camera& cam0, const Camera& cam1, const ImuNoise& noise,
                     const VioOptions& opt)
    : cam0_(cam0), opt_(opt), noise_(noise) {
  noise_.acc_noise *= opt.imu_noise_scale;
  noise_.gyro_noise *= opt.imu_noise_scale;
  noise_.acc_walk *= opt.imu_noise_scale;
  noise_.gyro_walk *= opt.imu_noise_scale;
  fe_ = std::make_unique<StereoFrontend>(cam0, cam1, opt.frontend);
}

void VioSystem::Initialize(const NominalState& x0, const Mat15& P0) {
  eskf_.Init(x0, P0, noise_);
  last_.t = x0.t;
  initialized_ = false;  // becomes true once the first IMU sample at/after t0 arrives
}

void VioSystem::AddImu(const ImuSample& m) {
  if (m.t <= eskf_.state().t) {
    // Remember the latest sample before the start so we can interpolate.
    last_ = m;
    return;
  }
  buf_.push_back(m);
}

void VioSystem::PropagateTo(double t) {
  size_t used = 0;
  for (; used < buf_.size() && buf_[used].t <= t; ++used) {
    const ImuSample& b = buf_[used];
    ImuSample a = last_;
    if (!initialized_) {
      // First interval: synthesize the sample at the state time.
      a = (last_.t < eskf_.state().t) ? Lerp(last_, b, eskf_.state().t) : b;
      initialized_ = true;
    }
    eskf_.Propagate(a, b);
    last_ = b;
  }
  buf_.erase(buf_.begin(), buf_.begin() + used);
  // Interpolate the remaining partial interval up to t.
  if (!buf_.empty() && last_.t < t && initialized_) {
    const ImuSample m = Lerp(last_, buf_.front(), t);
    eskf_.Propagate(last_, m);
    last_ = m;
  }
}

FrameLog VioSystem::AddStereo(double t, const cv::Mat& left, const cv::Mat& right) {
  FrameLog log;
  log.t = t;
  auto t0 = Clock::now();
  PropagateTo(t);
  auto t1 = Clock::now();

  Mat3 R12;
  Vec3 p12;
  eskf_.PredictedRelative(&R12, &p12);
  const Pose prior = BodyRelativeToCamera(R12, p12, cam0_.T_BC);
  log.fe = fe_->Process(left, right, prior);
  auto t2 = Clock::now();

  if (opt_.use_vision && log.fe.has_motion) {
    Mat3 R12m;
    Vec3 p12m;
    Mat6 cov;
    CameraRelativeToBody(log.fe.pnp.T_21, log.fe.pnp.cov, cam0_.T_BC, &R12m, &p12m, &cov);
    log.upd = eskf_.UpdateRelativePose(R12m, p12m, cov * opt_.meas_cov_scale, opt_.gate_chi2);
  }
  eskf_.Clone();
  auto t3 = Clock::now();
  log.x = eskf_.state();
  log.filter_ms = Ms(t0, t1) + Ms(t2, t3);
  log.total_ms = Ms(t0, t3);
  return log;
}

NominalState StaticInit(const std::vector<ImuSample>& w) {
  Vec3 a = Vec3::Zero(), g = Vec3::Zero();
  for (const auto& m : w) {
    a += m.acc;
    g += m.gyro;
  }
  a /= static_cast<double>(w.size());
  g /= static_cast<double>(w.size());
  NominalState x;
  x.t = w.back().t;
  // At rest the accelerometer reads +g along world up: R_WB a = |a| e_z.
  Mat3 R = RotationBetween(a.normalized(), Vec3::UnitZ());
  // Remove any yaw so the initial heading is zero (yaw is unobservable anyway).
  const Vec3 fwd = R * Vec3::UnitX();
  const double yaw = std::atan2(fwd.y(), fwd.x());
  x.R = Exp(Vec3(0, 0, -yaw)) * R;
  x.bg = g;
  return x;
}

}  // namespace vio
