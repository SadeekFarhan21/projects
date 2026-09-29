#include "vio/frontend.hpp"

#include <chrono>
#include <opencv2/features.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace vio {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
Vec2 ToVec(const cv::Point2f& p) { return Vec2(p.x, p.y); }
}  // namespace

StereoFrontend::StereoFrontend(const Camera& cam0, const Camera& cam1, const FrontendOptions& opt)
    : cam0_(cam0), cam1_(cam1), opt_(opt) {
  T_10_ = cam1.T_BC.inverse() * cam0.T_BC;
  baseline_ = T_10_.t.norm();
  opt_.pnp.focal = cam0.focal();
}

FrontendResult StereoFrontend::Process(const cv::Mat& left, const cv::Mat& right,
                                       const Pose& prior_T21) {
  FrontendResult out;
  auto t0 = Clock::now();

  // 1. Track previous left features into the current left image.
  if (!prev_left_.empty() && !tracks_.empty()) {
    std::vector<cv::Point2f> p0, p1, pb;
    p0.reserve(tracks_.size());
    for (const auto& tr : tracks_) p0.push_back(tr.px);
    // Seed the tracker with the IMU-predicted location of each 3D point.
    p1 = p0;
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (!tracks_[i].has_3d) continue;
      const Vec3 Xc = prior_T21 * tracks_[i].X;
      if (Xc.z() <= 0.1) continue;
      const Vec2 px = cam0_.Project(Xc);
      if (px.x() >= 0 && px.y() >= 0 && px.x() < left.cols && px.y() < left.rows)
        p1[i] = cv::Point2f(px.x(), px.y());
    }
    std::vector<uchar> st, stb;
    std::vector<float> err;
    const cv::Size win(opt_.klt_window, opt_.klt_window);
    const auto crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(prev_left_, left, p0, p1, st, err, win, opt_.klt_levels, crit,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    pb = p0;
    cv::calcOpticalFlowPyrLK(left, prev_left_, p1, pb, stb, err, win, opt_.klt_levels, crit,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<Track> kept;
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (!st[i] || !stb[i]) continue;
      if (cv::norm(pb[i] - p0[i]) > opt_.fb_check_px) continue;
      if (p1[i].x < 0 || p1[i].y < 0 || p1[i].x >= left.cols - 1 || p1[i].y >= left.rows - 1)
        continue;
      Track tr = tracks_[i];
      tr.px = p1[i];
      kept.push_back(tr);
    }
    tracks_ = std::move(kept);
  }
  auto t1 = Clock::now();
  out.timing.track_ms = Ms(t0, t1);

  // 2. PnP between the 3D points from frame k-1 and their pixels in frame k.
  if (!prev_left_.empty()) {
    std::vector<Vec3> X;
    std::vector<Mat3> C;
    std::vector<Vec2> x;
    std::vector<int> map;
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (!tracks_[i].has_3d) continue;
      X.push_back(tracks_[i].X);
      C.push_back(tracks_[i].cov);
      x.push_back(cam0_.Undistort(ToVec(tracks_[i].px)));
      map.push_back(static_cast<int>(i));
    }
    out.n_tracked = static_cast<int>(X.size());
    out.pnp = SolvePnpRansac(X, C, x, prior_T21, opt_.pnp);
    out.has_motion = out.pnp.ok;
    if (out.pnp.ok) {
      // Drop tracks that PnP flagged as outliers; they are likely bad KLT.
      std::vector<char> inl(tracks_.size(), 1);
      for (int m : map) inl[m] = 0;
      for (int k : out.pnp.inliers) inl[map[k]] = 1;
      std::vector<Track> kept;
      for (size_t i = 0; i < tracks_.size(); ++i)
        if (inl[i]) kept.push_back(tracks_[i]);
      tracks_ = std::move(kept);
    }
  }
  auto t2 = Clock::now();
  out.timing.pnp_ms = Ms(t1, t2);

  // 3. Top up features in empty image regions.
  Replenish(left);
  auto t3 = Clock::now();
  out.timing.detect_ms = Ms(t2, t3);

  // 4. Fresh stereo depth for every live feature (frame-to-frame VO).
  Stereo(left, right);
  auto t4 = Clock::now();
  out.timing.stereo_ms = Ms(t3, t4);

  out.n_features = static_cast<int>(tracks_.size());
  for (const auto& tr : tracks_) out.n_stereo += tr.has_3d;
  prev_left_ = left.clone();
  return out;
}

void StereoFrontend::Replenish(const cv::Mat& left) {
  const int need = opt_.max_features - static_cast<int>(tracks_.size());
  if (need <= 0) return;
  cv::Mat mask(left.size(), CV_8UC1, cv::Scalar(255));
  for (const auto& tr : tracks_)
    cv::circle(mask, tr.px, static_cast<int>(opt_.min_distance_px), cv::Scalar(0), -1);
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(left, corners, need, opt_.quality_level, opt_.min_distance_px, mask);
  for (const auto& c : corners) tracks_.push_back({next_id_++, c, false, Vec3::Zero(), Mat3::Zero()});
}

void StereoFrontend::Stereo(const cv::Mat& left, const cv::Mat& right) {
  if (tracks_.empty()) return;
  std::vector<cv::Point2f> pl, pr, pb;
  for (const auto& tr : tracks_) pl.push_back(tr.px);
  // Initial guess: shift by the disparity of a point ~3 m away.
  const float d0 = static_cast<float>(cam0_.focal() * baseline_ / 3.0);
  for (const auto& p : pl) pr.emplace_back(p.x - d0, p.y);
  std::vector<uchar> st, stb;
  std::vector<float> err;
  const cv::Size win(opt_.klt_window, opt_.klt_window);
  const auto crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(left, right, pl, pr, st, err, win, opt_.klt_levels, crit,
                           cv::OPTFLOW_USE_INITIAL_FLOW);
  pb = pl;
  cv::calcOpticalFlowPyrLK(right, left, pr, pb, stb, err, win, opt_.klt_levels, crit,
                           cv::OPTFLOW_USE_INITIAL_FLOW);
  const double f = cam0_.focal();
  for (size_t i = 0; i < tracks_.size(); ++i) {
    auto& tr = tracks_[i];
    tr.has_3d = false;
    if (!st[i] || !stb[i] || cv::norm(pb[i] - pl[i]) > opt_.fb_check_px) continue;
    const Vec2 x0 = cam0_.Undistort(ToVec(pl[i]));
    const Vec2 x1 = cam1_.Undistort(ToVec(pr[i]));
    Vec3 X;
    if (!Triangulate(x0, x1, T_10_, &X)) continue;
    if (X.z() < opt_.min_depth || X.z() > opt_.max_depth) continue;
    // Reject matches that violate the epipolar geometry.
    const double e0 = (ProjectNormalized(X) - x0).norm() * f;
    const double e1 = (ProjectNormalized(T_10_ * X) - x1).norm() * f;
    if (e0 > opt_.stereo_row_px || e1 > opt_.stereo_row_px) continue;
    tr.X = X;
    tr.cov = StereoPointCovariance(X, f, baseline_, opt_.pnp.sigma_px);
    tr.has_3d = true;
  }
}

}  // namespace vio
