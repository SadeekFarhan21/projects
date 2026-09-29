// IMU propagation against trajectories with closed-form solutions, plus a
// finite-difference check of the error-state transition matrix.
#include <gtest/gtest.h>

#include "vio/imu.hpp"
#include "vio/sim.hpp"

using namespace vio;

namespace {
constexpr double kDt = 0.005;  // 200 Hz, as on EuRoC

ImuSample Sample(double t, const Vec3& w, const Vec3& f) {
  ImuSample m;
  m.t = t;
  m.gyro = w;
  m.acc = f;
  return m;
}
}  // namespace

TEST(Imu, StaticLevelStaysPut) {
  NominalState x;
  const Vec3 f(0, 0, 9.81);  // specific force at rest, level
  for (int k = 0; k < 2000; ++k)
    PropagateNominal(x, Sample(k * kDt, Vec3::Zero(), f), Sample((k + 1) * kDt, Vec3::Zero(), f));
  EXPECT_NEAR(x.p.norm(), 0.0, 1e-9);
  EXPECT_NEAR(x.v.norm(), 0.0, 1e-9);
  EXPECT_NEAR(x.t, 10.0, 1e-9);
}

TEST(Imu, ConstantAccelerationStraightLine) {
  // p(t) = v0 t + a t^2 / 2 with a fixed, level attitude.
  NominalState x;
  x.v = Vec3(0.5, 0, 0);
  const Vec3 a(1.0, -0.5, 0.2);
  const Vec3 f = a - kGravity;
  const int N = 2000;
  for (int k = 0; k < N; ++k)
    PropagateNominal(x, Sample(k * kDt, Vec3::Zero(), f), Sample((k + 1) * kDt, Vec3::Zero(), f));
  const double T = N * kDt;
  const Vec3 p_ref = Vec3(0.5, 0, 0) * T + 0.5 * a * T * T;
  EXPECT_NEAR((x.p - p_ref).norm(), 0.0, 1e-8);
  EXPECT_NEAR((x.v - (Vec3(0.5, 0, 0) + a * T)).norm(), 0.0, 1e-9);
}

TEST(Imu, ConstantRateSpinMatchesExp) {
  NominalState x;
  const Vec3 w(0.1, -0.2, 0.3);
  // Keep the body in free fall so position is unaffected by the spin.
  const int N = 4000;
  for (int k = 0; k < N; ++k)
    PropagateNominal(x, Sample(k * kDt, w, Vec3::Zero()), Sample((k + 1) * kDt, w, Vec3::Zero()));
  EXPECT_NEAR((x.R - Exp(w * N * kDt)).norm(), 0.0, 1e-9);
  // Free fall: p = g t^2 / 2
  const double T = N * kDt;
  EXPECT_NEAR((x.p - 0.5 * kGravity * T * T).norm(), 0.0, 1e-7);
}

TEST(Imu, CoordinatedTurnCircle) {
  // Body flies a horizontal circle of radius r at yaw rate om, nose along the
  // velocity. Closed form: p = r (cos om t, sin om t, 0).
  const double r = 2.0, om = 0.5;
  NominalState x;
  x.p = Vec3(r, 0, 0);
  x.v = Vec3(0, r * om, 0);
  x.R = Exp(Vec3(0, 0, M_PI / 2));
  // In the body frame the centripetal acceleration always points along +y
  // (towards the center, to the left of the nose), constant magnitude r om^2.
  const Vec3 f_body = Vec3(0, r * om * om, 0) + Vec3(0, 0, 9.81);
  const Vec3 w(0, 0, om);
  const double T = 4 * M_PI / om;  // two full loops
  const int N = static_cast<int>(std::round(T / kDt));
  for (int k = 0; k < N; ++k)
    PropagateNominal(x, Sample(k * kDt, w, f_body), Sample((k + 1) * kDt, w, f_body));
  const double t = N * kDt;
  const Vec3 p_ref(r * std::cos(om * t), r * std::sin(om * t), 0);
  EXPECT_LT((x.p - p_ref).norm(), 1e-3);
  EXPECT_LT((x.v - Vec3(-r * om * std::sin(om * t), r * om * std::cos(om * t), 0)).norm(), 1e-4);
}

TEST(Imu, LissajousDeadReckoningWithIdealImu) {
  NominalState x;
  const auto s0 = sim::Lissajous(0);
  x.p = s0.p;
  x.v = s0.v;
  x.R = s0.R;
  ImuSample prev = sim::IdealImu(s0);
  const int N = 20 * 200;
  for (int k = 1; k <= N; ++k) {
    const ImuSample cur = sim::IdealImu(sim::Lissajous(k * kDt));
    PropagateNominal(x, prev, cur);
    prev = cur;
  }
  const auto sT = sim::Lissajous(N * kDt);
  EXPECT_LT((x.p - sT.p).norm(), 5e-3);
  EXPECT_LT(Log(sT.R.transpose() * x.R).norm(), 1e-5);
}

namespace {
// Error between two nominal states in the ESKF convention (a relative to b).
Eigen::Matrix<double, 15, 1> Diff(const NominalState& a, const NominalState& b) {
  Eigen::Matrix<double, 15, 1> e;
  e << a.p - b.p, a.v - b.v, Log(b.R.transpose() * a.R), a.bg - b.bg, a.ba - b.ba;
  return e;
}
NominalState Plus(const NominalState& x, const Eigen::Matrix<double, 15, 1>& d) {
  NominalState y = x;
  y.p += d.segment<3>(0);
  y.v += d.segment<3>(3);
  y.R = x.R * Exp(d.segment<3>(6));
  y.bg += d.segment<3>(9);
  y.ba += d.segment<3>(12);
  return y;
}
}  // namespace

TEST(Imu, ErrorStateTransitionMatchesFiniteDifference) {
  NominalState x;
  x.R = Exp(Vec3(0.2, -0.3, 0.9));
  x.v = Vec3(1.0, -0.5, 0.3);
  x.bg = Vec3(0.01, -0.02, 0.005);
  x.ba = Vec3(0.1, 0.05, -0.08);
  // Chain 20 steps (0.1 s) of a rich motion so every block is exercised.
  std::vector<ImuSample> seq;
  for (int k = 0; k <= 20; ++k) {
    const double t = k * kDt;
    seq.push_back(Sample(t, Vec3(0.5 * std::sin(3 * t), 0.8, -0.4 + t), Vec3(1.0 + t, -2.0, 9.0 + std::cos(5 * t))));
  }
  Mat15 Phi = Mat15::Identity(), Phi_k, Q;
  NominalState xn = x;
  for (size_t k = 0; k + 1 < seq.size(); ++k) {
    PropagateImu(xn, seq[k], seq[k + 1], ImuNoise{}, kGravity, &Phi_k, &Q);
    Phi = Phi_k * Phi;
  }
  auto run = [&](const NominalState& s) {
    NominalState y = s;
    for (size_t k = 0; k + 1 < seq.size(); ++k) PropagateNominal(y, seq[k], seq[k + 1]);
    return y;
  };
  const double h = 1e-6;
  Mat15 num;
  for (int i = 0; i < 15; ++i) {
    Eigen::Matrix<double, 15, 1> d = Eigen::Matrix<double, 15, 1>::Zero();
    d(i) = h;
    num.col(i) = (Diff(run(Plus(x, d)), xn) - Diff(run(Plus(x, -d)), xn)) / (2 * h);
  }
  EXPECT_LT((num - Phi).cwiseAbs().maxCoeff(), 2e-4) << "numeric\n" << num << "\nanalytic\n" << Phi;
}

TEST(Imu, StaticCovarianceGrowthMatchesClosedForm) {
  // With only white noise, at rest and level:
  //   var(theta_i) = sg^2 T,  var(v_x) = sa^2 T + g^2 sg^2 T^3 / 3
  ImuNoise n;
  n.gyro_walk = 0;
  n.acc_walk = 0;
  NominalState x;
  Mat15 P = Mat15::Zero(), Phi, Q;
  const Vec3 f(0, 0, 9.81);
  const int N = 2000;
  for (int k = 0; k < N; ++k) {
    PropagateImu(x, Sample(k * kDt, Vec3::Zero(), f), Sample((k + 1) * kDt, Vec3::Zero(), f), n, kGravity, &Phi, &Q);
    P = Phi * P * Phi.transpose() + Q;
  }
  const double T = N * kDt, g = 9.81;
  const double sg2 = n.gyro_noise * n.gyro_noise, sa2 = n.acc_noise * n.acc_noise;
  EXPECT_NEAR(P(idx::TH, idx::TH) / (sg2 * T), 1.0, 0.01);
  EXPECT_NEAR(P(idx::V, idx::V) / (sa2 * T + g * g * sg2 * T * T * T / 3), 1.0, 0.02);
  // Vertical velocity does not couple to attitude at first order.
  EXPECT_NEAR(P(idx::V + 2, idx::V + 2) / (sa2 * T), 1.0, 0.02);
}
