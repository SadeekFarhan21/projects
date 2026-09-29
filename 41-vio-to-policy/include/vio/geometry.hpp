// Camera model, stereo triangulation and PnP with RANSAC.
#pragma once

#include <Eigen/Core>
#include <random>
#include <vector>

#include "vio/eskf.hpp"
#include "vio/so3.hpp"

namespace vio {

using Vec2 = Eigen::Vector2d;
using Mat4 = Eigen::Matrix4d;

struct Pose {
  Mat3 R = Mat3::Identity();
  Vec3 t = Vec3::Zero();
  Vec3 operator*(const Vec3& x) const { return R * x + t; }
  Pose operator*(const Pose& o) const { return {R * o.R, R * o.t + t}; }
  Pose inverse() const { return {R.transpose(), -R.transpose() * t}; }
  static Pose FromMatrix(const Mat4& T) { return {T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>()}; }
};

// Pinhole camera with radial-tangential distortion (EuRoC calibration).
struct Camera {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  double k1 = 0, k2 = 0, p1 = 0, p2 = 0;
  int width = 0, height = 0;
  Pose T_BC;  // body_from_camera

  // Distort a normalized point and map it to pixels.
  Vec2 Project(const Vec3& X) const;
  // Inverse of the distortion model by fixed-point iteration, pixels -> normalized.
  Vec2 Undistort(const Vec2& px) const;
  double focal() const { return 0.5 * (fx + fy); }
};

inline Vec2 ProjectNormalized(const Vec3& X) { return Vec2(X.x() / X.z(), X.y() / X.z()); }

// Linear (DLT) two-view triangulation. T_10 maps points from cam0 into cam1.
// x0, x1 are normalized image coordinates. Returns false for points behind
// either camera.
bool Triangulate(const Vec2& x0, const Vec2& x1, const Pose& T_10, Vec3* X0);

// Approximate 3x3 covariance of a stereo-triangulated point in cam0:
// sigma_lat along the image plane and a depth sigma that grows with z^2.
Mat3 StereoPointCovariance(const Vec3& X0, double focal, double baseline, double sigma_px);

struct PnpOptions {
  int ransac_iters = 100;
  int sample_size = 4;
  double inlier_px = 2.0;     // reprojection threshold in pixels
  double focal = 458.0;        // to convert pixel thresholds into normalized units
  double sigma_px = 1.0;       // measurement noise of a tracked keypoint
  int min_inliers = 12;
  int gn_iters = 8;
  unsigned seed = 7;
};

struct PnpResult {
  bool ok = false;
  Pose T_21;               // maps points in frame 1 into frame 2
  Mat6 cov = Mat6::Zero(); // over [phi, dt], left perturbation: R = Exp(phi) R_hat, t = t_hat + dt
  std::vector<int> inliers;
  double rms_px = 0.0;
};

// Estimate T_21 from 3D points in frame 1 (with covariance) and normalized
// observations in frame 2. RANSAC hypotheses come from small-sample
// Gauss-Newton seeded at `prior`; the winner is refined with a weighted
// Gauss-Newton that folds the 3D point uncertainty into each residual.
PnpResult SolvePnpRansac(const std::vector<Vec3>& X1, const std::vector<Mat3>& cov_X1,
                         const std::vector<Vec2>& x2, const Pose& prior, const PnpOptions& opt);

// Convert a camera relative pose T_c2c1 (with PnP covariance) into the body
// relative pose used by the filter: R12 = R_b1^T R_b2, p12 = R_b1^T (p_b2 - p_b1).
void CameraRelativeToBody(const Pose& T_c2c1, const Mat6& cov_c, const Pose& T_BC, Mat3* R12,
                          Vec3* p12, Mat6* cov_b);

// Inverse direction: body relative pose (clone -> current) to T_c2c1. Used to
// turn the IMU prediction into a PnP prior.
Pose BodyRelativeToCamera(const Mat3& R12, const Vec3& p12, const Pose& T_BC);

}  // namespace vio
