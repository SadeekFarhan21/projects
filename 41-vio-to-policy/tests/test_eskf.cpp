#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <random>

#include "vio/eskf.hpp"
#include "vio/sim.hpp"

using namespace vio;

namespace {

Eskf MakeFilter() {
  NominalState x;
  x.p = Vec3(1, 2, 3);
  x.v = Vec3(0.3, -0.1, 0.2);
  x.R = Exp(Vec3(0.1, -0.2, 0.7));
  Eskf f;
  f.Init(x, sim::DefaultP0(), ImuNoise{});
  // Move the current pose away from the clone so the Jacobian is non-trivial.
  Vec21 dx = Vec21::Zero();
  dx.segment<3>(idx::P) = Vec3(0.4, -0.3, 0.1);
  dx.segment<3>(idx::TH) = Vec3(0.05, 0.1, -0.2);
  f.Inject(dx);
  return f;
}

// Measurement function in error coordinates: z(dx) for the relative pose.
Vec6 Measure(const Eskf& f0, const Vec21& d) {
  Eskf f = f0;
  Mat3 R12h;
  Vec3 p12h;
  f0.PredictedRelative(&R12h, &p12h);
  // Apply the perturbation without the covariance reset side effects.
  NominalState& x = f.mutable_state();
  x.p += d.segment<3>(idx::P);
  x.R = x.R * Exp(d.segment<3>(idx::TH));
  const Mat3 R1 = f0.clone_R() * Exp(d.segment<3>(idx::CTH));
  const Vec3 p1 = f0.clone_p() + d.segment<3>(idx::CP);
  const Mat3 R12 = R1.transpose() * x.R;
  const Vec3 p12 = R1.transpose() * (x.p - p1);
  Vec6 z;
  z << Log(R12h.transpose() * R12), p12 - p12h;
  return z;
}

}  // namespace

TEST(Eskf, RelativePoseJacobianMatchesFiniteDifference) {
  const Eskf f = MakeFilter();
  const auto H = f.RelativePoseJacobian();
  const double h = 1e-6;
  Eigen::Matrix<double, 6, kStateDim> num;
  for (int i = 0; i < kStateDim; ++i) {
    Vec21 d = Vec21::Zero();
    d(i) = h;
    num.col(i) = (Measure(f, d) - Measure(f, -d)) / (2 * h);
  }
  EXPECT_LT((num - H).cwiseAbs().maxCoeff(), 1e-6) << "numeric\n" << num << "\nanalytic\n" << H;
}

TEST(Eskf, CloneCopiesPoseCovariance) {
  Eskf f = MakeFilter();
  f.Clone();
  const Mat21& P = f.cov();
  EXPECT_TRUE((P.block<3, 3>(idx::CP, idx::CP).isApprox(P.block<3, 3>(idx::P, idx::P))));
  EXPECT_TRUE((P.block<3, 3>(idx::CTH, idx::CTH).isApprox(P.block<3, 3>(idx::TH, idx::TH))));
  EXPECT_TRUE((P.block<3, 3>(idx::CP, idx::V).isApprox(P.block<3, 3>(idx::P, idx::V))));
  EXPECT_TRUE(f.clone_R().isApprox(f.state().R));
}

TEST(Eskf, UpdateShrinksCovarianceAndStaysSymmetric) {
  Eskf f = MakeFilter();
  f.Clone();
  // Let uncertainty grow for a bit, then fuse a zero-motion measurement.
  ImuSample a, b;
  a.acc = b.acc = f.state().R.transpose() * Vec3(0, 0, 9.81);
  for (int k = 0; k < 20; ++k) {
    a.t = k * 0.005;
    b.t = (k + 1) * 0.005;
    f.Propagate(a, b);
  }
  const double before = f.cov().trace();
  Mat3 R12;
  Vec3 p12;
  f.PredictedRelative(&R12, &p12);
  const auto res = f.UpdateRelativePose(R12, p12, Mat6::Identity() * 1e-6, -1);
  EXPECT_TRUE(res.accepted);
  EXPECT_LT(f.cov().trace(), before);
  EXPECT_NEAR((f.cov() - f.cov().transpose()).norm(), 0.0, 1e-12);
  Eigen::SelfAdjointEigenSolver<Mat21> es(f.cov());
  EXPECT_GT(es.eigenvalues().minCoeff(), -1e-12);
}

TEST(Eskf, GateRejectsGrossOutlier) {
  Eskf f = MakeFilter();
  f.Clone();
  Mat3 R12;
  Vec3 p12;
  f.PredictedRelative(&R12, &p12);
  const auto res = f.UpdateRelativePose(R12, p12 + Vec3(5, 0, 0), Mat6::Identity() * 1e-4, 22.46);
  EXPECT_FALSE(res.accepted);
  EXPECT_GT(res.nis, 22.46);
}

// Filter consistency: averaged NEES over Monte Carlo runs on a synthetic
// trajectory with known noise must sit inside the chi-square band.
TEST(EskfConsistency, NeesWithinChiSquareBounds) {
  const int runs = 25;
  sim::McRunConfig cfg;
  cfg.duration = 30.0;
  cfg.P0 = sim::DefaultP0();
  std::vector<std::vector<sim::McStep>> all;
  for (int r = 0; r < runs; ++r) {
    cfg.seed = 100 + r;
    all.push_back(sim::RunMonteCarlo(cfg));
  }
  const size_t T = all[0].size();
  double mean_core = 0, mean_pose = 0;
  int in_core = 0;
  const double lo = sim::Chi2Quantile(0.025, 15.0 * runs) / runs;
  const double hi = sim::Chi2Quantile(0.975, 15.0 * runs) / runs;
  for (size_t k = 0; k < T; ++k) {
    double c = 0, p = 0;
    for (const auto& run : all) {
      c += run[k].nees_core;
      p += run[k].nees_pose;
    }
    c /= runs;
    p /= runs;
    mean_core += c / T;
    mean_pose += p / T;
    in_core += (c >= lo && c <= hi);
  }
  // Time-averaged ANEES close to the state dimension ...
  EXPECT_GT(mean_core, 0.8 * 15);
  EXPECT_LT(mean_core, 1.2 * 15);
  EXPECT_GT(mean_pose, 0.8 * 6);
  EXPECT_LT(mean_pose, 1.2 * 6);
  // ... and inside the 95 percent band most of the time.
  EXPECT_GT(static_cast<double>(in_core) / T, 0.85);
}

TEST(EskfConsistency, OverconfidentFilterIsDetected) {
  // Same experiment but the filter believes the IMU is 5x quieter than it is.
  const int runs = 10;
  sim::McRunConfig cfg;
  cfg.duration = 20.0;
  cfg.P0 = sim::DefaultP0();
  cfg.filter_imu_scale = 0.2;
  double mean_core = 0;
  size_t n = 0;
  for (int r = 0; r < runs; ++r) {
    cfg.seed = 500 + r;
    for (const auto& s : sim::RunMonteCarlo(cfg)) {
      mean_core += s.nees_core;
      ++n;
    }
  }
  mean_core /= n;
  EXPECT_GT(mean_core, 2.0 * 15);
}

TEST(Chi2, QuantileApproximation) {
  // Reference values from standard tables.
  EXPECT_NEAR(sim::Chi2Quantile(0.95, 6), 12.592, 0.05);
  // Wilson-Hilferty is about 1 percent off in the far tail at low dof.
  EXPECT_NEAR(sim::Chi2Quantile(0.999, 6), 22.458, 0.3);
  EXPECT_NEAR(sim::Chi2Quantile(0.025, 15), 6.262, 0.05);
}
