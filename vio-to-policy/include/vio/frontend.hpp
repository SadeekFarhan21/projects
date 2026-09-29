// Stereo visual odometry frontend: KLT tracking, stereo triangulation, PnP.
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "vio/geometry.hpp"

namespace vio {

struct FrontendOptions {
  int max_features = 200;
  double min_distance_px = 20.0;
  double quality_level = 0.01;
  int klt_window = 21;
  int klt_levels = 3;
  double fb_check_px = 1.0;       // forward-backward consistency threshold
  double stereo_row_px = 3.0;     // max reprojection error of a stereo match
  double min_depth = 0.3, max_depth = 25.0;
  PnpOptions pnp;
};

struct FrontendTiming {
  double track_ms = 0, pnp_ms = 0, detect_ms = 0, stereo_ms = 0;
};

struct FrontendResult {
  bool has_motion = false;   // true when a PnP estimate between frames exists
  PnpResult pnp;             // T_21 = cam(k) <- cam(k-1)
  int n_tracked = 0;         // features tracked from k-1 that carry 3D
  int n_features = 0;        // features alive after replenishing
  int n_stereo = 0;          // features with a fresh stereo 3D point
  FrontendTiming timing;
};

class StereoFrontend {
 public:
  StereoFrontend(const Camera& cam0, const Camera& cam1, const FrontendOptions& opt);

  // Process a rectified-or-not grayscale stereo pair. `prior_T21` is the
  // IMU-predicted cam(k) <- cam(k-1) transform, used to seed PnP.
  FrontendResult Process(const cv::Mat& left, const cv::Mat& right, const Pose& prior_T21);

 private:
  struct Track {
    int id;
    cv::Point2f px;      // left image pixel
    bool has_3d = false;
    Vec3 X;              // in current left camera frame
    Mat3 cov;
  };
  void Replenish(const cv::Mat& left);
  void Stereo(const cv::Mat& left, const cv::Mat& right);

  Camera cam0_, cam1_;
  Pose T_10_;  // cam1 <- cam0
  double baseline_;
  FrontendOptions opt_;
  cv::Mat prev_left_;
  std::vector<Track> tracks_;
  int next_id_ = 0;
};

}  // namespace vio
