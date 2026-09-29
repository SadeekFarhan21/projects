#include "vio/geometry.hpp"

#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <numeric>

namespace vio {

Vec2 Camera::Project(const Vec3& X) const {
  const double x = X.x() / X.z(), y = X.y() / X.z();
  const double r2 = x * x + y * y;
  const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
  const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
  return Vec2(fx * xd + cx, fy * yd + cy);
}

Vec2 Camera::Undistort(const Vec2& px) const {
  const double xd = (px.x() - cx) / fx, yd = (px.y() - cy) / fy;
  double x = xd, y = yd;
  // Fixed-point iteration; converges in a handful of steps for EuRoC lenses.
  for (int i = 0; i < 20; ++i) {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    const double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    const double xn = (xd - dx) / radial, yn = (yd - dy) / radial;
    if (std::abs(xn - x) + std::abs(yn - y) < 1e-12) {
      x = xn;
      y = yn;
      break;
    }
    x = xn;
    y = yn;
  }
  return Vec2(x, y);
}

bool Triangulate(const Vec2& x0, const Vec2& x1, const Pose& T_10, Vec3* X0) {
  // Rows of x cross (P X) = 0 for P0 = [I | 0], P1 = [R | t].
  Eigen::Matrix<double, 4, 4> A;
  Eigen::Matrix<double, 3, 4> P0 = Eigen::Matrix<double, 3, 4>::Zero();
  P0.leftCols<3>().setIdentity();
  Eigen::Matrix<double, 3, 4> P1;
  P1.leftCols<3>() = T_10.R;
  P1.rightCols<1>() = T_10.t;
  A.row(0) = x0.x() * P0.row(2) - P0.row(0);
  A.row(1) = x0.y() * P0.row(2) - P0.row(1);
  A.row(2) = x1.x() * P1.row(2) - P1.row(0);
  A.row(3) = x1.y() * P1.row(2) - P1.row(1);
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d h = svd.matrixV().col(3);
  if (std::abs(h(3)) < 1e-12) return false;
  *X0 = h.head<3>() / h(3);
  const Vec3 X1 = T_10 * (*X0);
  return X0->z() > 0.0 && X1.z() > 0.0;
}

Mat3 StereoPointCovariance(const Vec3& X0, double focal, double baseline, double sigma_px) {
  const double z = X0.z();
  const double sigma_lat = z * sigma_px / focal;
  // Disparity noise is the difference of two keypoint errors.
  const double sigma_depth = z * z * std::sqrt(2.0) * sigma_px / (focal * baseline);
  const Vec3 u = X0.normalized();
  const Mat3 uut = u * u.transpose();
  return sigma_lat * sigma_lat * (Mat3::Identity() - uut) + sigma_depth * sigma_depth * uut;
}

namespace {

using Mat26 = Eigen::Matrix<double, 2, 6>;

// Residual r = x_obs - pi(R X + t) and its Jacobian w.r.t. [phi, dt] under the
// left perturbation R <- Exp(phi) R, t <- t + dt. Also returns d r / d X.
bool Residual(const Pose& T, const Vec3& X, const Vec2& x, Vec2* r, Mat26* J,
              Eigen::Matrix<double, 2, 3>* JX) {
  const Vec3 Xc = T * X;
  if (Xc.z() < 1e-3) return false;
  const double iz = 1.0 / Xc.z();
  *r = x - Vec2(Xc.x() * iz, Xc.y() * iz);
  Eigen::Matrix<double, 2, 3> Jp;
  Jp << iz, 0.0, -Xc.x() * iz * iz, 0.0, iz, -Xc.y() * iz * iz;
  if (J) {
    // d(Xc)/d(phi) = -[R X]x, d(Xc)/d(dt) = I; residual carries a minus sign.
    J->leftCols<3>() = -Jp * (-skew(T.R * X));
    J->rightCols<3>() = -Jp;
  }
  if (JX) *JX = -Jp * T.R;
  return true;
}

// Gauss-Newton over a subset of correspondences. If `cov_X` is non-null each
// residual is whitened by sigma^2 I + JX cov_X JX^T. Returns false on failure.
bool GaussNewton(const std::vector<Vec3>& X1, const std::vector<Mat3>* cov_X,
                 const std::vector<Vec2>& x2, const std::vector<int>& ids, double sigma_n,
                 int iters, Pose* T, Mat6* info_out) {
  for (int it = 0; it < iters; ++it) {
    Mat6 Hm = Mat6::Zero();
    Vec6 b = Vec6::Zero();
    int used = 0;
    for (int i : ids) {
      Vec2 r;
      Mat26 J;
      Eigen::Matrix<double, 2, 3> JX;
      if (!Residual(*T, X1[i], x2[i], &r, &J, &JX)) continue;
      Eigen::Matrix2d S = sigma_n * sigma_n * Eigen::Matrix2d::Identity();
      if (cov_X) S += JX * (*cov_X)[i] * JX.transpose();
      const Eigen::Matrix2d W = S.inverse();
      // r(x + d) ~ r + J d ; minimize ||r + J d||_W
      Hm += J.transpose() * W * J;
      b += J.transpose() * W * r;
      ++used;
    }
    if (used < 3) return false;
    const Vec6 d = Hm.ldlt().solve(-b);
    if (!d.allFinite()) return false;
    T->R = Orthonormalize(Exp(d.head<3>()) * T->R);
    T->t += d.tail<3>();
    if (info_out) *info_out = Hm;
    if (d.norm() < 1e-10) break;
  }
  if (info_out) {
    // Recompute information at the final estimate.
    Mat6 Hm = Mat6::Zero();
    for (int i : ids) {
      Vec2 r;
      Mat26 J;
      Eigen::Matrix<double, 2, 3> JX;
      if (!Residual(*T, X1[i], x2[i], &r, &J, &JX)) continue;
      Eigen::Matrix2d S = sigma_n * sigma_n * Eigen::Matrix2d::Identity();
      if (cov_X) S += JX * (*cov_X)[i] * JX.transpose();
      Hm += J.transpose() * S.inverse() * J;
    }
    *info_out = Hm;
  }
  return true;
}

std::vector<int> CountInliers(const std::vector<Vec3>& X1, const std::vector<Vec2>& x2,
                              const Pose& T, double thr_n) {
  std::vector<int> in;
  const double thr2 = thr_n * thr_n;
  for (size_t i = 0; i < X1.size(); ++i) {
    const Vec3 Xc = T * X1[i];
    if (Xc.z() < 1e-3) continue;
    const Vec2 r = x2[i] - ProjectNormalized(Xc);
    if (r.squaredNorm() < thr2) in.push_back(static_cast<int>(i));
  }
  return in;
}

}  // namespace

PnpResult SolvePnpRansac(const std::vector<Vec3>& X1, const std::vector<Mat3>& cov_X1,
                         const std::vector<Vec2>& x2, const Pose& prior, const PnpOptions& opt) {
  PnpResult res;
  const int n = static_cast<int>(X1.size());
  if (n < std::max(opt.sample_size, opt.min_inliers)) return res;
  const double thr_n = opt.inlier_px / opt.focal;
  const double sigma_n = opt.sigma_px / opt.focal;

  Pose best = prior;
  std::vector<int> best_in = CountInliers(X1, x2, prior, thr_n);
  std::mt19937 rng(opt.seed);
  std::vector<int> all(n);
  std::iota(all.begin(), all.end(), 0);
  for (int it = 0; it < opt.ransac_iters; ++it) {
    // Partial Fisher-Yates for a random sample without replacement.
    for (int k = 0; k < opt.sample_size; ++k) {
      std::uniform_int_distribution<int> U(k, n - 1);
      std::swap(all[k], all[U(rng)]);
    }
    std::vector<int> sample(all.begin(), all.begin() + opt.sample_size);
    Pose T = prior;
    if (!GaussNewton(X1, nullptr, x2, sample, sigma_n, 6, &T, nullptr)) continue;
    auto in = CountInliers(X1, x2, T, thr_n);
    if (in.size() > best_in.size()) {
      best_in = std::move(in);
      best = T;
      // Early exit once the inlier ratio is overwhelming.
      if (best_in.size() > 0.9 * n) break;
    }
  }
  if (static_cast<int>(best_in.size()) < opt.min_inliers) return res;

  // Refine on inliers, re-select inliers once, refine again with full weights.
  Pose T = best;
  Mat6 info;
  if (!GaussNewton(X1, &cov_X1, x2, best_in, sigma_n, opt.gn_iters, &T, &info)) return res;
  best_in = CountInliers(X1, x2, T, thr_n);
  if (static_cast<int>(best_in.size()) < opt.min_inliers) return res;
  if (!GaussNewton(X1, &cov_X1, x2, best_in, sigma_n, opt.gn_iters, &T, &info)) return res;

  Eigen::LDLT<Mat6> ldlt(info);
  if (ldlt.info() != Eigen::Success) return res;
  res.cov = ldlt.solve(Mat6::Identity());
  res.cov = 0.5 * (res.cov + res.cov.transpose());
  double ss = 0.0;
  for (int i : best_in) ss += (x2[i] - ProjectNormalized(T * X1[i])).squaredNorm();
  res.rms_px = std::sqrt(ss / best_in.size()) * opt.focal;
  res.T_21 = T;
  res.inliers = std::move(best_in);
  res.ok = true;
  return res;
}

Pose BodyRelativeToCamera(const Mat3& R12, const Vec3& p12, const Pose& T_BC) {
  const Pose T_b1b2{R12, p12};
  const Pose T_c1c2 = T_BC.inverse() * T_b1b2 * T_BC;
  return T_c1c2.inverse();
}

void CameraRelativeToBody(const Pose& T_c2c1, const Mat6& cov_c, const Pose& T_BC, Mat3* R12,
                          Vec3* p12, Mat6* cov_b) {
  auto to_body = [&](const Pose& T21) {
    return T_BC * T21.inverse() * T_BC.inverse();  // T_b1b2
  };
  const Pose Tb = to_body(T_c2c1);
  *R12 = Tb.R;
  *p12 = Tb.t;
  // Numerical Jacobian of the body residual w.r.t. the PnP perturbation.
  Mat6 J;
  const double h = 1e-6;
  for (int k = 0; k < 6; ++k) {
    Vec6 d = Vec6::Zero();
    d(k) = h;
    Pose Tp = T_c2c1, Tm = T_c2c1;
    Tp.R = Exp(d.head<3>()) * T_c2c1.R;
    Tp.t = T_c2c1.t + d.tail<3>();
    Tm.R = Exp(-d.head<3>()) * T_c2c1.R;
    Tm.t = T_c2c1.t - d.tail<3>();
    const Pose Bp = to_body(Tp), Bm = to_body(Tm);
    Vec6 col;
    col.head<3>() = (Log(Tb.R.transpose() * Bp.R) - Log(Tb.R.transpose() * Bm.R)) / (2 * h);
    col.tail<3>() = (Bp.t - Bm.t) / (2 * h);
    J.col(k) = col;
  }
  *cov_b = J * cov_c * J.transpose();
  *cov_b = 0.5 * (*cov_b + cov_b->transpose());
}

}  // namespace vio
