// SO(3) and quaternion helpers.
//
// Conventions used everywhere in this project:
//   * Quaternions are Hamilton, stored as Eigen::Quaterniond (w, x, y, z).
//   * R_WB maps a vector expressed in body coordinates into world coordinates.
//   * Orientation error is a LOCAL (right) perturbation: R = R_hat * Exp(dtheta).
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

namespace vio {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Quaterniond;

inline Mat3 skew(const Vec3& v) {
  Mat3 m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}

inline Vec3 vee(const Mat3& m) { return Vec3(m(2, 1), m(0, 2), m(1, 0)); }

// Rodrigues formula with a Taylor expansion near zero.
inline Mat3 Exp(const Vec3& w) {
  const double th2 = w.squaredNorm();
  const Mat3 W = skew(w);
  if (th2 < 1e-12) {
    return Mat3::Identity() + W + 0.5 * W * W;
  }
  const double th = std::sqrt(th2);
  return Mat3::Identity() + (std::sin(th) / th) * W + ((1.0 - std::cos(th)) / th2) * W * W;
}

// Inverse of Exp. Valid on the whole group, including angles close to pi.
inline Vec3 Log(const Mat3& R) {
  const double c = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
  const double th = std::acos(c);
  if (th < 1e-6) {
    // First order: R ~ I + [w]x
    return 0.5 * vee(R - R.transpose());
  }
  if (M_PI - th < 1e-4) {
    // Near pi the antisymmetric part vanishes; use the quaternion route,
    // which is numerically stable there.
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
  }
  return (th / (2.0 * std::sin(th))) * vee(R - R.transpose());
}

// Quaternion exponential of a rotation vector (unit quaternion out).
inline Quat QExp(const Vec3& w) {
  const double th = w.norm();
  if (th < 1e-12) {
    Quat q(1.0, 0.5 * w.x(), 0.5 * w.y(), 0.5 * w.z());
    return q.normalized();
  }
  const Vec3 a = w / th;
  const double s = std::sin(0.5 * th);
  return Quat(std::cos(0.5 * th), s * a.x(), s * a.y(), s * a.z());
}

inline Vec3 QLog(const Quat& q_in) {
  Quat q = q_in.normalized();
  if (q.w() < 0.0) q.coeffs() *= -1.0;  // shortest path
  const double vn = q.vec().norm();
  if (vn < 1e-12) return 2.0 * q.vec();
  const double th = 2.0 * std::atan2(vn, q.w());
  return th * q.vec() / vn;
}

// Right Jacobian of SO(3): Exp(w + dw) ~ Exp(w) Exp(Jr(w) dw).
inline Mat3 RightJacobian(const Vec3& w) {
  const double th2 = w.squaredNorm();
  const Mat3 W = skew(w);
  if (th2 < 1e-10) return Mat3::Identity() - 0.5 * W + (1.0 / 6.0) * W * W;
  const double th = std::sqrt(th2);
  return Mat3::Identity() - ((1.0 - std::cos(th)) / th2) * W +
         ((th - std::sin(th)) / (th2 * th)) * W * W;
}

// Box-plus / box-minus for rotations under the local perturbation convention.
inline Mat3 BoxPlus(const Mat3& R, const Vec3& d) { return R * Exp(d); }
inline Vec3 BoxMinus(const Mat3& R_a, const Mat3& R_b) { return Log(R_b.transpose() * R_a); }

// Re-orthonormalize a rotation matrix that has picked up round-off drift.
inline Mat3 Orthonormalize(const Mat3& R) {
  Quat q(R);
  return q.normalized().toRotationMatrix();
}

// Rotation that takes unit vector a onto unit vector b (minimal rotation).
inline Mat3 RotationBetween(const Vec3& a, const Vec3& b) {
  return Quat::FromTwoVectors(a, b).toRotationMatrix();
}

}  // namespace vio
