// Error-state EKF with one stochastically cloned pose.
//
// Error state (21):
//   [0:3)  dp    position
//   [3:6)  dv    velocity
//   [6:9)  dth   attitude, local:  R = R_hat Exp(dth)
//   [9:12) dbg   gyro bias
//   [12:15) dba  accel bias
//   [15:18) dth1 attitude of the cloned pose (previous camera frame)
//   [18:21) dp1  position of the cloned pose
//
// Stereo VO produces a relative body pose between the clone (frame k-1) and
// the current frame k. Cloning keeps the cross-covariance between the two
// poses, so a relative measurement is fused without double counting.
#pragma once

#include <Eigen/Core>

#include "vio/imu.hpp"

namespace vio {

constexpr int kStateDim = 21;
namespace idx {
constexpr int CTH = 15;
constexpr int CP = 18;
}  // namespace idx

using Mat21 = Eigen::Matrix<double, kStateDim, kStateDim>;
using Vec21 = Eigen::Matrix<double, kStateDim, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

struct UpdateResult {
  bool accepted = false;
  double nis = 0.0;  // normalized innovation squared, chi2 with 6 dof if consistent
  Vec6 residual = Vec6::Zero();
};

class Eskf {
 public:
  Eskf() = default;

  void Init(const NominalState& x0, const Mat15& P0, const ImuNoise& noise, const Vec3& g = kGravity);

  // Propagate core state and covariance between two consecutive IMU samples.
  void Propagate(const ImuSample& a, const ImuSample& b);

  // Replace the clone with the current pose (marginalizes the old clone).
  void Clone();

  // Fuse a relative pose measurement expressed as body frame k-1 -> body k:
  //   R12 = R1^T R2,   p12 = R1^T (p2 - p1)
  // cov is 6x6 over [dtheta (local on R12), dp12].
  // Measurements with NIS above `gate` are rejected (gate <= 0 disables).
  UpdateResult UpdateRelativePose(const Mat3& R12_meas, const Vec3& p12_meas, const Mat6& cov,
                                  double gate);

  // Predicted relative pose from clone to current state.
  void PredictedRelative(Mat3* R12, Vec3* p12) const;

  const NominalState& state() const { return x_; }
  NominalState& mutable_state() { return x_; }
  const Mat21& cov() const { return P_; }
  Mat21& mutable_cov() { return P_; }
  const Mat3& clone_R() const { return R1_; }
  const Vec3& clone_p() const { return p1_; }
  bool has_clone() const { return has_clone_; }

  // Measurement Jacobian, exposed for testing.
  Eigen::Matrix<double, 6, kStateDim> RelativePoseJacobian() const;

  // Apply an error-state correction (inject + covariance reset).
  void Inject(const Vec21& dx);

 private:
  NominalState x_;
  Mat3 R1_ = Mat3::Identity();
  Vec3 p1_ = Vec3::Zero();
  bool has_clone_ = false;
  Mat21 P_ = Mat21::Zero();
  ImuNoise noise_;
  Vec3 g_ = kGravity;
};

}  // namespace vio
