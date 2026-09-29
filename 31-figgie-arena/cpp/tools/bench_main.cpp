// Raw C++ throughput: full games per second for a few lineups (single
// thread, no Python), and the cost of one exact posterior update.
// Usage: figgie_bench [deals] [out.json]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "figgie/arena.hpp"

using namespace figgie;

int main(int argc, char** argv) {
  int deals = argc > 1 ? std::atoi(argv[1]) : 500;
  std::string out = argc > 2 ? argv[2] : "";
  std::string json = "{\"lineups\": [";
  const std::vector<std::vector<std::string>> lineups = {
      {"random", "random", "random", "random"},
      {"passive", "passive", "taker", "taker"},
      {"bayes", "passive", "taker", "random"},
      {"bayes", "bayes", "bayes", "bayes"}};
  Config cfg;
  for (size_t i = 0; i < lineups.size(); ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto r = run_games(lineups[i], deals, 7, cfg, 4, true, false, 1);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double trades = 0;
    for (int t : r.n_trades) trades += t;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "%s{\"lineup\": \"%s,%s,%s,%s\", \"games\": %d, \"seconds\": %.3f, \"games_per_sec\": %.1f, "
                  "\"ticks_per_sec\": %.0f, \"mean_trades\": %.2f}",
                  i ? ", " : "", lineups[i][0].c_str(), lineups[i][1].c_str(), lineups[i][2].c_str(),
                  lineups[i][3].c_str(), r.n_games, sec, r.n_games / sec, r.n_games * cfg.ticks / sec,
                  trades / r.n_games);
    json += buf;
  }
  // Posterior cost with realistic constraints.
  Counts hand{4, 3, 2, 1};
  std::vector<Counts> mins = {Counts{2, 1, 0, 1}, Counts{0, 2, 1, 0}, Counts{1, 0, 2, 1}};
  const int reps = 2000;
  double sink = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < reps; ++k) sink += config_posterior(hand, mins, 10)[k % 12];
  double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
  char buf[256];
  std::snprintf(buf, sizeof buf, "], \"posterior_update_us\": %.1f, \"sink\": %.3f}", us, sink);
  json += buf;
  std::puts(json.c_str());
  if (!out.empty()) std::ofstream(out) << json << "\n";
}
