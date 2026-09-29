// Invariant fuzzer: plays many random games and checks every rule invariant
// after every action. Usage: figgie_fuzz [games] [seed] [threads] [out.json] [bayes 0|1]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "figgie/arena.hpp"

int main(int argc, char** argv) {
  uint64_t games = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 100000;
  uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0;
  int threads = argc > 3 ? std::atoi(argv[3]) : 4;
  std::string out = argc > 4 ? argv[4] : "";
  bool bayes = argc > 5 ? std::atoi(argv[5]) != 0 : true;
  auto t0 = std::chrono::steady_clock::now();
  auto r = figgie::fuzz(games, seed, threads, bayes);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  char buf[1024];
  std::snprintf(buf, sizeof buf,
                "{\"games\": %llu, \"actions\": %llu, \"trades\": %llu, \"rejected_actions\": %llu, "
                "\"invariant_violations\": %llu, \"seed\": %llu, \"threads\": %d, \"bayes_seats\": %s, \"seconds\": %.2f}",
                (unsigned long long)r.games, (unsigned long long)r.actions, (unsigned long long)r.trades,
                (unsigned long long)r.rejected, (unsigned long long)r.violations, (unsigned long long)seed, threads, bayes ? "true" : "false", sec);
  std::puts(buf);
  if (!r.first_violation.empty()) std::fprintf(stderr, "first violation: %s\n", r.first_violation.c_str());
  if (!out.empty()) std::ofstream(out) << buf << "\n";
  return r.violations == 0 ? 0 : 1;
}
