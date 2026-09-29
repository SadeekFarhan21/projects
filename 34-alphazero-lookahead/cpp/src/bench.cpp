// Micro benchmarks: solver time by game stage and pure MCTS rollout speed.
// Usage: c4_bench [positions_per_stage]
// Prints CSV rows: kind,stage,count,mean_ms,max_ms,mean_nodes
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "c4/pure_mcts.hpp"
#include "c4/solver.hpp"

using namespace c4;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const int per_stage = argc > 1 ? std::atoi(argv[1]) : 20;
  std::mt19937_64 rng(1234);
  std::printf("kind,stage,count,mean_ms,max_ms,mean_nodes\n");
  for (int stage : {8, 12, 16, 20, 24, 28, 32}) {
    Solver s(24);
    double total = 0, worst = 0, nodes = 0;
    int done = 0;
    while (done < per_stage) {
      Position p;
      bool ok = true;
      while (p.moves() < stage) {
        p.play(random_move(p, rng));
        if (p.is_terminal()) { ok = false; break; }
      }
      if (!ok) continue;
      s.reset();
      auto t0 = Clock::now();
      (void)s.analyze(p);
      double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      total += ms; worst = std::max(worst, ms); nodes += static_cast<double>(s.node_count());
      ++done;
    }
    std::printf("solver_analyze,%d,%d,%.3f,%.3f,%.0f\n", stage, done, total / done, worst, nodes / done);
    std::fflush(stdout);
  }
  for (int rollouts : {100, 1000, 10000}) {
    PureMCTS m(rollouts, 7);
    auto t0 = Clock::now();
    const int reps = 20;
    for (int i = 0; i < reps; ++i) (void)m.choose_move(Position::from_moves("4453"));
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;
    std::printf("pure_mcts_move,%d,%d,%.3f,%.3f,0\n", rollouts, reps, ms, ms);
  }
  return 0;
}
