// Monte Carlo NEES experiment on the synthetic Lissajous trajectory.
//
//   sim_nees --runs 50 --duration 60 --out results/nees
// Writes <out>_per_step.csv (average NEES over runs at each camera time) and
// <out>_summary.json.
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "vio/sim.hpp"

using namespace vio;

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
  const int runs = a.count("runs") ? std::stoi(a["runs"]) : 50;
  const double duration = a.count("duration") ? std::stod(a["duration"]) : 60.0;
  const std::string out = a.count("out") ? a["out"] : "results/nees";
  // Optional mis-tuning: filter believes IMU noise is this factor times the truth.
  const double filter_imu_scale = a.count("filter-imu-scale") ? std::stod(a["filter-imu-scale"]) : 1.0;

  sim::McRunConfig cfg;
  cfg.duration = duration;
  cfg.P0 = sim::DefaultP0();
  cfg.filter_imu_scale = filter_imu_scale;
  std::vector<std::vector<sim::McStep>> all;
  for (int r = 0; r < runs; ++r) {
    cfg.seed = 1000 + r;
    all.push_back(sim::RunMonteCarlo(cfg));
  }

  const size_t T = all.front().size();
  FILE* f = std::fopen((out + "_per_step.csv").c_str(), "w");
  std::fprintf(f, "t,anees_core,anees_pose,mean_pos_err\n");
  double sum_core = 0, sum_pose = 0;
  int in_core = 0, in_pose = 0;
  const double lo15 = sim::Chi2Quantile(0.025, 15.0 * runs) / runs;
  const double hi15 = sim::Chi2Quantile(0.975, 15.0 * runs) / runs;
  const double lo6 = sim::Chi2Quantile(0.025, 6.0 * runs) / runs;
  const double hi6 = sim::Chi2Quantile(0.975, 6.0 * runs) / runs;
  for (size_t k = 0; k < T; ++k) {
    double c = 0, p = 0, e = 0;
    for (const auto& run : all) {
      c += run[k].nees_core;
      p += run[k].nees_pose;
      e += run[k].pos_err;
    }
    c /= runs;
    p /= runs;
    e /= runs;
    sum_core += c;
    sum_pose += p;
    in_core += (c >= lo15 && c <= hi15);
    in_pose += (p >= lo6 && p <= hi6);
    std::fprintf(f, "%.3f,%.4f,%.4f,%.5f\n", all.front()[k].t, c, p, e);
  }
  std::fclose(f);
  FILE* j = std::fopen((out + "_summary.json").c_str(), "w");
  std::fprintf(j,
               "{\n  \"runs\": %d,\n  \"filter_imu_scale\": %.3f,\n  \"duration_s\": %.1f,\n  \"steps\": %zu,\n"
               "  \"anees_core_mean\": %.4f,\n  \"anees_core_dof\": 15,\n"
               "  \"core_bounds_95\": [%.4f, %.4f],\n  \"core_frac_in_bounds\": %.4f,\n"
               "  \"anees_pose_mean\": %.4f,\n  \"anees_pose_dof\": 6,\n"
               "  \"pose_bounds_95\": [%.4f, %.4f],\n  \"pose_frac_in_bounds\": %.4f\n}\n",
               runs, filter_imu_scale, duration, T, sum_core / T, lo15, hi15, double(in_core) / T, sum_pose / T, lo6, hi6,
               double(in_pose) / T);
  std::fclose(j);
  std::printf("ANEES core %.3f (15 dof, bounds %.2f..%.2f, %.1f%% in), pose %.3f (6 dof, bounds %.2f..%.2f, %.1f%% in)\n",
              sum_core / T, lo15, hi15, 100.0 * in_core / T, sum_pose / T, lo6, hi6, 100.0 * in_pose / T);
  return 0;
}
