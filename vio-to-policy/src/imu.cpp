#include "vio/imu.hpp"

namespace vio {

void PropagateImu(NominalState& x, const ImuSample& a, const ImuSample& b, const ImuNoise& n,
                  const Vec3& g, Mat15* Phi, Mat15* Qd) {
  const double dt = b.t - a.t;
  if (dt <= 0.0) {
    if (Phi) Phi->setIdentity();
    if (Qd) Qd->setZero();
    return;
  }
  // Bias-corrected midpoint measurements.
  const Vec3 w = 0.5 * (a.gyro + b.gyro) - x.bg;
  const Vec3 fa = a.acc - x.ba;
  const Vec3 fb = b.acc - x.ba;

  const Mat3 R0 = x.R;
  const Mat3 dR = Exp(w * dt);
  const Mat3 R1 = R0 * dR;
  // Midpoint world acceleration (trapezoid on the rotated specific force).
  const Vec3 acc_w = 0.5 * (R0 * fa + R1 * fb) + g;

  x.p += x.v * dt + 0.5 * acc_w * dt * dt;
  x.v += acc_w * dt;
  x.R = Orthonormalize(R1);
  x.t = b.t;

  if (!Phi && !Qd) return;

  // Error dynamics (local rotation error, R = R_hat Exp(dth)):
  //   d(dp)/dt  = dv
  //   d(dv)/dt  = -R [f]x dth - R dba - R na
  //   d(dth)/dt = -[w]x dth - dbg - ng
  //   d(dbg)/dt = nbg,  d(dba)/dt = nba
  // Discretized to second order in dt, with the exact rotation block.
  const Vec3 f = 0.5 * (fa + fb);
  const Mat3 Rm = R0 * Exp(0.5 * w * dt);  // attitude at the interval midpoint
  const Mat3 I = Mat3::Identity();

  Mat15 F = Mat15::Zero();
  F.block<3, 3>(idx::P, idx::V) = I;
  F.block<3, 3>(idx::V, idx::TH) = -Rm * skew(f);
  F.block<3, 3>(idx::V, idx::BA) = -Rm;
  F.block<3, 3>(idx::TH, idx::TH) = -skew(w);
  F.block<3, 3>(idx::TH, idx::BG) = -I;

  Mat15 P = Mat15::Identity() + F * dt + 0.5 * F * F * dt * dt;
  P.block<3, 3>(idx::TH, idx::TH) = dR.transpose();
  if (Phi) *Phi = P;

  if (Qd) {
    // G maps [na, ng, nbg, nba] into the error state.
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
    G.block<3, 3>(idx::V, 0) = -Rm;
    G.block<3, 3>(idx::TH, 3) = -I;
    G.block<3, 3>(idx::BG, 6) = I;
    G.block<3, 3>(idx::BA, 9) = I;
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block<3, 3>(0, 0) = n.acc_noise * n.acc_noise * I;
    Qc.block<3, 3>(3, 3) = n.gyro_noise * n.gyro_noise * I;
    Qc.block<3, 3>(6, 6) = n.gyro_walk * n.gyro_walk * I;
    Qc.block<3, 3>(9, 9) = n.acc_walk * n.acc_walk * I;
    // Van Loan to first order: Qd = Phi G Qc G^T Phi^T dt.
    const Mat15 Qg = G * Qc * G.transpose() * dt;
    *Qd = P * Qg * P.transpose();
    *Qd = 0.5 * (*Qd + Qd->transpose());
  }
}

}  // namespace vio
