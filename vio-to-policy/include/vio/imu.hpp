// IMU samples, nominal state and strapdown propagation.
#pragma once

#include <Eigen/Core>
#include <vector>

#include "vio/so3.hpp"

namespace vio {

struct ImuSample {
  double t = 0.0;         // seconds
  Vec3 gyro = Vec3::Zero();  // rad/s, body frame
  Vec3 acc = Vec3::Zero();   // m/s^2 specific force, body frame
};

// Continuous-time noise densities (EuRoC sensor.yaml units).
struct ImuNoise {
  double gyro_noise = 1.6968e-4;   // rad/s/sqrt(Hz)
  double acc_noise = 2.0e-3;       // m/s^2/sqrt(Hz)
  double gyro_walk = 1.9393e-5;    // rad/s^2/sqrt(Hz)
  double acc_walk = 3.0e-3;        // m/s^3/sqrt(Hz)
};

// Nominal (large-signal) state. The EKF tracks a 15-dim error around it.
struct NominalState {
  double t = 0.0;
  Vec3 p = Vec3::Zero();     // position of body in world
  Vec3 v = Vec3::Zero();     // velocity of body in world
  Mat3 R = Mat3::Identity(); // R_WB
  Vec3 bg = Vec3::Zero();    // gyro bias
  Vec3 ba = Vec3::Zero();    // accel bias
};

inline const Vec3 kGravity(0.0, 0.0, -9.81);

// Error-state index layout (15 core states).
namespace idx {
constexpr int P = 0;
constexpr int V = 3;
constexpr int TH = 6;
constexpr int BG = 9;
constexpr int BA = 12;
constexpr int CORE = 15;
}  // namespace idx

using Mat15 = Eigen::Matrix<double, 15, 15>;

// Propagate the nominal state from sample a to sample b (b.t > a.t) with a
// midpoint scheme, and return the discrete error-state transition Phi and
// process noise Qd for that interval.
void PropagateImu(NominalState& x, const ImuSample& a, const ImuSample& b, const ImuNoise& n,
                  const Vec3& g, Mat15* Phi, Mat15* Qd);

// Nominal-only propagation (no Jacobians); used for prediction and tests.
inline void PropagateNominal(NominalState& x, const ImuSample& a, const ImuSample& b,
                             const Vec3& g = kGravity) {
  PropagateImu(x, a, b, ImuNoise{}, g, nullptr, nullptr);
}

}  // namespace vio
