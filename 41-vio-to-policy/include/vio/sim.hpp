// Synthetic trajectories, noisy IMU and relative pose measurements with known
// statistics. Used by the unit tests and the NEES Monte Carlo experiment.
#pragma once

#include <random>
#include <vector>

#include "vio/eskf.hpp"

namespace vio::sim {

struct TruthSample {
  double t;
  Vec3 p, v, a;  // world position, velocity, acceleration
  Mat3 R;        // R_WB
  Vec3 w;        // body angular rate
};

// Smooth 3D Lissajous path with oscillating roll, pitch and yaw.
TruthSample Lissajous(double t);

// Ideal IMU reading (specific force and body rate) for a truth sample.
ImuSample IdealImu(const TruthSample& s, const Vec3& g = kGravity);

struct MeasNoise {
  double rot_sigma = 0.005;   // rad, per axis
  double pos_sigma = 0.01;    // m, per axis
  Mat6 Cov() const;
};

struct McRunConfig {
  double duration = 60.0;
  double imu_rate = 200.0;
  double cam_rate = 20.0;
  ImuNoise imu;
  MeasNoise meas;
  Mat15 P0;
  unsigned seed = 1;
  // The filter is told the IMU densities are this factor times the truth
  // (1.0 = correctly tuned). Used to show that NEES detects mis-tuning.
  double filter_imu_scale = 1.0;
};

struct McStep {
  double t;
  double nees_core;  // 15 dof
  double nees_pose;  // 6 dof (position + attitude)
  double pos_err;    // meters
};

// One Monte Carlo run: sample an initial error from P0, simulate noisy IMU with
// random-walk biases and noisy relative-pose measurements, run the ESKF and
// return NEES at every camera time.
std::vector<McStep> RunMonteCarlo(const McRunConfig& cfg);

// Default initial covariance used by the experiments.
Mat15 DefaultP0();

// Chi-square quantile (Wilson-Hilferty approximation, good for dof >= 3).
double Chi2Quantile(double p, double dof);

}  // namespace vio::sim
