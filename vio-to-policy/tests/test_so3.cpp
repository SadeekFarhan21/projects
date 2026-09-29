#include <gtest/gtest.h>

#include <random>

#include "vio/so3.hpp"

using namespace vio;

namespace {
std::mt19937 rng(42);
Vec3 RandomRotVec(double max_angle) {
  std::uniform_real_distribution<double> U(-1, 1), A(0, max_angle);
  Vec3 axis(U(rng), U(rng), U(rng));
  return axis.normalized() * A(rng);
}
}  // namespace

TEST(SO3, ExpOfZeroIsIdentity) { EXPECT_TRUE(Exp(Vec3::Zero()).isApprox(Mat3::Identity())); }

TEST(SO3, SkewMatchesCrossProduct) {
  const Vec3 a(1, -2, 3), b(0.5, 4, -1);
  EXPECT_TRUE((skew(a) * b).isApprox(a.cross(b)));
  EXPECT_TRUE(vee(skew(a)).isApprox(a));
}

TEST(SO3, ExpIsOrthonormalWithUnitDeterminant) {
  for (int i = 0; i < 200; ++i) {
    const Mat3 R = Exp(RandomRotVec(M_PI));
    EXPECT_NEAR((R.transpose() * R - Mat3::Identity()).norm(), 0.0, 1e-12);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
  }
}

TEST(SO3, ExpMatchesEigenAngleAxis) {
  for (int i = 0; i < 200; ++i) {
    const Vec3 w = RandomRotVec(M_PI);
    const Mat3 ref = Eigen::AngleAxisd(w.norm(), w.normalized()).toRotationMatrix();
    EXPECT_NEAR((Exp(w) - ref).norm(), 0.0, 1e-12);
  }
}

TEST(SO3, LogInvertsExp) {
  for (int i = 0; i < 500; ++i) {
    const Vec3 w = RandomRotVec(M_PI - 1e-3);
    EXPECT_NEAR((Log(Exp(w)) - w).norm(), 0.0, 1e-9) << w.transpose();
  }
}

TEST(SO3, LogSmallAngles) {
  const Vec3 w(1e-9, -2e-9, 3e-10);
  EXPECT_NEAR((Log(Exp(w)) - w).norm(), 0.0, 1e-15);
}

TEST(SO3, LogNearPi) {
  const Vec3 axis = Vec3(1, 2, -1).normalized();
  for (double eps : {1e-3, 1e-5, 1e-7}) {
    const Vec3 w = (M_PI - eps) * axis;
    const Vec3 l = Log(Exp(w));
    // Either w or its antipode (same rotation at exactly pi) is acceptable;
    // below pi the answer is unique.
    EXPECT_NEAR((Exp(l) - Exp(w)).norm(), 0.0, 1e-9);
    EXPECT_NEAR(l.norm(), M_PI - eps, 1e-6);
  }
}

TEST(SO3, QuaternionExpMatchesMatrixExp) {
  for (int i = 0; i < 200; ++i) {
    const Vec3 w = RandomRotVec(M_PI);
    EXPECT_NEAR((QExp(w).toRotationMatrix() - Exp(w)).norm(), 0.0, 1e-12);
    EXPECT_NEAR((QLog(QExp(w)) - w).norm(), 0.0, 1e-9);
  }
}

TEST(SO3, QuaternionDoubleCoverGivesSameLog) {
  const Vec3 w(0.3, -0.2, 0.1);
  Quat q = QExp(w);
  Quat mq(-q.w(), -q.x(), -q.y(), -q.z());
  EXPECT_NEAR((QLog(mq) - w).norm(), 0.0, 1e-12);
}

TEST(SO3, HamiltonCompositionMatchesMatrixProduct) {
  for (int i = 0; i < 100; ++i) {
    const Vec3 a = RandomRotVec(M_PI), b = RandomRotVec(M_PI);
    const Quat qa = QExp(a), qb = QExp(b);
    EXPECT_NEAR(((qa * qb).toRotationMatrix() - Exp(a) * Exp(b)).norm(), 0.0, 1e-12);
    // Rotating a vector: q v q* equals R v.
    const Vec3 v(0.3, 1.2, -0.7);
    EXPECT_NEAR((qa * v - Exp(a) * v).norm(), 0.0, 1e-12);
  }
}

TEST(SO3, RightJacobianMatchesFiniteDifference) {
  for (int i = 0; i < 50; ++i) {
    const Vec3 w = RandomRotVec(2.5);
    const Mat3 Jr = RightJacobian(w);
    const double h = 1e-7;
    for (int k = 0; k < 3; ++k) {
      Vec3 d = Vec3::Zero();
      d(k) = h;
      // Exp(w + d) = Exp(w) Exp(Jr d)
      const Vec3 col = Log(Exp(w).transpose() * Exp(w + d)) / h;
      EXPECT_NEAR((col - Jr.col(k)).norm(), 0.0, 1e-6);
    }
  }
}

TEST(SO3, BoxPlusMinusAreInverse) {
  for (int i = 0; i < 100; ++i) {
    const Mat3 R = Exp(RandomRotVec(M_PI));
    const Vec3 d = RandomRotVec(1.0);
    EXPECT_NEAR((BoxMinus(BoxPlus(R, d), R) - d).norm(), 0.0, 1e-9);
  }
}

TEST(SO3, RotationBetweenAlignsVectors) {
  const Vec3 a = Vec3(0.2, -0.5, 0.9).normalized(), b = Vec3(0, 0, 1);
  EXPECT_NEAR((RotationBetween(a, b) * a - b).norm(), 0.0, 1e-12);
}
