#include <gtest/gtest.h>

#include <random>

#include "vio/geometry.hpp"

using namespace vio;

namespace {

Camera EurocCam0() {
  Camera c;
  c.fx = 458.654;
  c.fy = 457.296;
  c.cx = 367.215;
  c.cy = 248.375;
  c.k1 = -0.28340811;
  c.k2 = 0.07395907;
  c.p1 = 0.00019359;
  c.p2 = 1.76187114e-05;
  c.width = 752;
  c.height = 480;
  return c;
}

// cam1 <- cam0 for a ~11 cm horizontal baseline with a small rotation.
Pose StereoExtrinsic() { return {Exp(Vec3(0.002, -0.003, 0.001)), Vec3(-0.11, 0.0005, 0.001)}; }

struct Scene {
  std::vector<Vec3> X;  // in frame 1
};

Scene RandomScene(std::mt19937& rng, int n) {
  std::uniform_real_distribution<double> U(-1, 1), Z(1.5, 8.0);
  Scene s;
  for (int i = 0; i < n; ++i) {
    const double z = Z(rng);
    s.X.emplace_back(U(rng) * 0.6 * z, U(rng) * 0.4 * z, z);
  }
  return s;
}

}  // namespace

TEST(Camera, UndistortInvertsDistortion) {
  const Camera c = EurocCam0();
  for (double u = 20; u < 740; u += 60)
    for (double v = 20; v < 470; v += 50) {
      const Vec2 n = c.Undistort(Vec2(u, v));
      const Vec2 px = c.Project(Vec3(n.x(), n.y(), 1.0));
      EXPECT_NEAR((px - Vec2(u, v)).norm(), 0.0, 1e-6) << u << "," << v;
    }
}

TEST(Stereo, TriangulationIsExactWithoutNoise) {
  std::mt19937 rng(1);
  const Pose T10 = StereoExtrinsic();
  const Scene s = RandomScene(rng, 100);
  for (const auto& X : s.X) {
    Vec3 Xh;
    ASSERT_TRUE(Triangulate(ProjectNormalized(X), ProjectNormalized(T10 * X), T10, &Xh));
    EXPECT_NEAR((Xh - X).norm(), 0.0, 1e-8);
  }
}

TEST(Stereo, DepthCovarianceMatchesMonteCarlo) {
  std::mt19937 rng(2);
  std::normal_distribution<double> N(0, 1);
  const Pose T10 = StereoExtrinsic();
  const double f = 458.0, sigma_px = 0.5;
  for (double z : {2.0, 5.0}) {
    const Vec3 X(0.3, -0.2, z);
    std::vector<double> depths;
    for (int k = 0; k < 4000; ++k) {
      const Vec2 x0 = ProjectNormalized(X) + Vec2(N(rng), N(rng)) * sigma_px / f;
      const Vec2 x1 = ProjectNormalized(T10 * X) + Vec2(N(rng), N(rng)) * sigma_px / f;
      Vec3 Xh;
      if (Triangulate(x0, x1, T10, &Xh)) depths.push_back(Xh.z());
    }
    double m = 0, v = 0;
    for (double d : depths) m += d;
    m /= depths.size();
    for (double d : depths) v += (d - m) * (d - m);
    const double emp = std::sqrt(v / depths.size());
    const Mat3 C = StereoPointCovariance(X, f, T10.t.norm(), sigma_px);
    const double pred = std::sqrt(C(2, 2));
    EXPECT_NEAR(emp / pred, 1.0, 0.15) << "z=" << z << " emp " << emp << " pred " << pred;
  }
}

TEST(Pnp, RecoversPoseWithOutliers) {
  std::mt19937 rng(3);
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> U(-0.6, 0.6);
  const Scene s = RandomScene(rng, 150);
  const Pose T_true{Exp(Vec3(0.02, -0.05, 0.03)), Vec3(0.08, -0.02, 0.15)};
  const double f = 458.0;
  std::vector<Vec3> X;
  std::vector<Mat3> C;
  std::vector<Vec2> x;
  int n_out = 0;
  for (size_t i = 0; i < s.X.size(); ++i) {
    X.push_back(s.X[i]);
    C.push_back(Mat3::Identity() * 1e-4);
    if (i % 10 < 3) {  // 30 percent gross outliers
      x.emplace_back(U(rng), U(rng));
      ++n_out;
    } else {
      x.push_back(ProjectNormalized(T_true * s.X[i]) + Vec2(N(rng), N(rng)) * 0.5 / f);
    }
  }
  // Prior is off by several degrees and 10 cm.
  const Pose prior{Exp(Vec3(0.05, 0.0, -0.04)) * T_true.R, T_true.t + Vec3(0.1, 0.0, -0.05)};
  PnpOptions opt;
  opt.focal = f;
  const PnpResult r = SolvePnpRansac(X, C, x, prior, opt);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(Log(T_true.R.transpose() * r.T_21.R).norm() * 180 / M_PI, 0.2);
  EXPECT_LT((r.T_21.t - T_true.t).norm(), 0.01);
  EXPECT_GE(static_cast<int>(r.inliers.size()), 100);
  EXPECT_LE(static_cast<int>(r.inliers.size()), 150 - n_out + 3);
}

TEST(Pnp, CovarianceIsCalibrated) {
  // Pose NEES over many noisy trials should average to 6 when both the pixel
  // noise and the stereo point noise match the model.
  std::mt19937 rng(4);
  std::normal_distribution<double> N(0, 1);
  const double f = 458.0, sigma_px = 1.0, baseline = 0.11;
  const Pose T_true{Exp(Vec3(0.01, 0.02, -0.01)), Vec3(0.05, 0.0, 0.1)};
  PnpOptions opt;
  opt.focal = f;
  opt.sigma_px = sigma_px;
  opt.inlier_px = 50;  // no outliers here; keep everything
  double nees = 0;
  const int trials = 300;
  for (int k = 0; k < trials; ++k) {
    const Scene s = RandomScene(rng, 80);
    std::vector<Vec3> X;
    std::vector<Mat3> C;
    std::vector<Vec2> x;
    for (const auto& Xt : s.X) {
      const Mat3 cov = StereoPointCovariance(Xt, f, baseline, sigma_px);
      const Mat3 L = cov.llt().matrixL();
      X.push_back(Xt + L * Vec3(N(rng), N(rng), N(rng)));
      C.push_back(cov);
      x.push_back(ProjectNormalized(T_true * Xt) + Vec2(N(rng), N(rng)) * sigma_px / f);
    }
    const PnpResult r = SolvePnpRansac(X, C, x, T_true, opt);
    ASSERT_TRUE(r.ok);
    Vec6 e;
    e << Log(T_true.R * r.T_21.R.transpose()), T_true.t - r.T_21.t;
    nees += e.dot(r.cov.ldlt().solve(e));
  }
  nees /= trials;
  // Linearized point covariance makes this approximate; accept 6 +- 25 percent.
  EXPECT_GT(nees, 4.5);
  EXPECT_LT(nees, 7.5);
}

TEST(Frames, CameraBodyRelativeRoundTrip) {
  const Pose T_BC{Exp(Vec3(0.1, -1.5, 0.3)), Vec3(-0.02, -0.06, 0.01)};
  const Mat3 R12 = Exp(Vec3(0.02, 0.03, -0.1));
  const Vec3 p12(0.1, -0.05, 0.02);
  const Pose T21 = BodyRelativeToCamera(R12, p12, T_BC);
  Mat3 R;
  Vec3 p;
  Mat6 cov;
  CameraRelativeToBody(T21, Mat6::Identity() * 1e-4, T_BC, &R, &p, &cov);
  EXPECT_NEAR((R - R12).norm(), 0.0, 1e-12);
  EXPECT_NEAR((p - p12).norm(), 0.0, 1e-12);
  // Rotation part of a rigid change of frame preserves isotropic rotation noise.
  const double tr = cov.topLeftCorner<3, 3>().trace();
  EXPECT_NEAR(tr, 3e-4, 1e-8);
}
