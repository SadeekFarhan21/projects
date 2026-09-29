#include "vio/eskf.hpp"

#include <Eigen/Cholesky>

namespace vio {

void Eskf::Init(const NominalState& x0, const Mat15& P0, const ImuNoise& noise, const Vec3& g) {
  x_ = x0;
  noise_ = noise;
  g_ = g;
  P_.setZero();
  P_.topLeftCorner<15, 15>() = P0;
  has_clone_ = false;
  Clone();
}

void Eskf::Propagate(const ImuSample& a, const ImuSample& b) {
  Mat15 Phi, Qd;
  PropagateImu(x_, a, b, noise_, g_, &Phi, &Qd);
  // Core block and core/clone cross terms. The clone does not move.
  P_.topLeftCorner<15, 15>() = Phi * P_.topLeftCorner<15, 15>() * Phi.transpose() + Qd;
  P_.block<15, 6>(0, 15) = Phi * P_.block<15, 6>(0, 15);
  P_.block<6, 15>(15, 0) = P_.block<15, 6>(0, 15).transpose();
}

void Eskf::Clone() {
  R1_ = x_.R;
  p1_ = x_.p;
  has_clone_ = true;
  // New clone error equals current pose error: rows/cols copied from core.
  Eigen::Matrix<double, kStateDim, kStateDim> J = Mat21::Zero();
  J.topLeftCorner<15, 15>().setIdentity();
  J.block<3, 3>(idx::CTH, idx::TH).setIdentity();
  J.block<3, 3>(idx::CP, idx::P).setIdentity();
  P_ = J * P_ * J.transpose();
  P_ = 0.5 * (P_ + P_.transpose());
}

void Eskf::PredictedRelative(Mat3* R12, Vec3* p12) const {
  *R12 = R1_.transpose() * x_.R;
  *p12 = R1_.transpose() * (x_.p - p1_);
}

Eigen::Matrix<double, 6, kStateDim> Eskf::RelativePoseJacobian() const {
  Mat3 R12;
  Vec3 p12;
  PredictedRelative(&R12, &p12);
  Eigen::Matrix<double, 6, kStateDim> H = Eigen::Matrix<double, 6, kStateDim>::Zero();
  // Rotation rows: Log(R12_hat^T R12) ~ dth2 - R12_hat^T dth1
  H.block<3, 3>(0, idx::TH) = Mat3::Identity();
  H.block<3, 3>(0, idx::CTH) = -R12.transpose();
  // Translation rows: p12 ~ p12_hat + R1^T (dp2 - dp1) + [p12_hat]x dth1
  H.block<3, 3>(3, idx::P) = R1_.transpose();
  H.block<3, 3>(3, idx::CP) = -R1_.transpose();
  H.block<3, 3>(3, idx::CTH) = skew(p12);
  return H;
}

UpdateResult Eskf::UpdateRelativePose(const Mat3& R12_meas, const Vec3& p12_meas,
                                      const Mat6& cov, double gate) {
  UpdateResult res;
  Mat3 R12;
  Vec3 p12;
  PredictedRelative(&R12, &p12);
  Vec6 r;
  r.head<3>() = Log(R12.transpose() * R12_meas);
  r.tail<3>() = p12_meas - p12;
  res.residual = r;

  const auto H = RelativePoseJacobian();
  const Mat6 S = H * P_ * H.transpose() + cov;
  Eigen::LLT<Mat6> llt(S);
  if (llt.info() != Eigen::Success) return res;
  res.nis = r.dot(llt.solve(r));
  if (gate > 0.0 && res.nis > gate) return res;

  const Eigen::Matrix<double, kStateDim, 6> PHt = P_ * H.transpose();
  const Eigen::Matrix<double, kStateDim, 6> K = llt.solve(PHt.transpose()).transpose();
  const Vec21 dx = K * r;
  // Joseph form keeps P symmetric positive semidefinite.
  const Mat21 IKH = Mat21::Identity() - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * cov * K.transpose();
  Inject(dx);
  res.accepted = true;
  return res;
}

void Eskf::Inject(const Vec21& dx) {
  x_.p += dx.segment<3>(idx::P);
  x_.v += dx.segment<3>(idx::V);
  x_.R = Orthonormalize(x_.R * Exp(dx.segment<3>(idx::TH)));
  x_.bg += dx.segment<3>(idx::BG);
  x_.ba += dx.segment<3>(idx::BA);
  R1_ = Orthonormalize(R1_ * Exp(dx.segment<3>(idx::CTH)));
  p1_ += dx.segment<3>(idx::CP);
  // Covariance reset for the attitude blocks: G = I - [dth/2]x.
  Mat21 G = Mat21::Identity();
  G.block<3, 3>(idx::TH, idx::TH) -= skew(0.5 * dx.segment<3>(idx::TH));
  G.block<3, 3>(idx::CTH, idx::CTH) -= skew(0.5 * dx.segment<3>(idx::CTH));
  P_ = G * P_ * G.transpose();
  P_ = 0.5 * (P_ + P_.transpose());
}

}  // namespace vio
