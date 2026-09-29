#include <gtest/gtest.h>

#include <cmath>

#include "figgie/arena.hpp"

using namespace figgie;

TEST(Arena, DeterministicAcrossThreadCounts) {
  Config cfg;
  cfg.ticks = 60;
  std::vector<std::string> lineup{"bayes", "passive", "taker", "random"};
  auto a = run_games(lineup, 40, 11, cfg, 4, true, true, 1);
  auto b = run_games(lineup, 40, 11, cfg, 4, true, true, 2);
  EXPECT_EQ(a.pnl, b.pnl);
  EXPECT_EQ(a.brier, b.brier);
  EXPECT_EQ(a.invariant_violations, 0u) << a.first_violation;
}

TEST(Arena, ZeroSumAndRotation) {
  Config cfg;
  cfg.ticks = 80;
  auto r = run_games({"bayes", "random", "random", "random"}, 50, 3, cfg, 2, true, true, 2);
  ASSERT_EQ(r.n_games, 200);
  for (int g = 0; g < r.n_games; ++g) {
    double s = 0;
    for (int i = 0; i < 4; ++i) s += r.pnl[g * 4 + i];
    EXPECT_NEAR(s, 0.0, 1e-9);
    EXPECT_EQ(r.deal[g], g / 4);
    EXPECT_EQ(r.rotation[g], g % 4);
  }
  EXPECT_EQ(r.invariant_violations, 0u) << r.first_violation;
  for (double b : r.brier) {
    EXPECT_GE(b, 0.0);
    EXPECT_LE(b, 2.0);
  }
}

TEST(Arena, FuzzHasNoViolations) {
  auto r = fuzz(20000, 99, 2, /*allow_bayes=*/false);
  EXPECT_EQ(r.games, 20000u);
  EXPECT_GT(r.trades, 0u);
  EXPECT_GT(r.rejected, 0u);  // chaos players do send invalid actions
  EXPECT_EQ(r.violations, 0u) << r.first_violation;
  auto rb = fuzz(500, 7, 2, /*allow_bayes=*/true);
  EXPECT_EQ(rb.violations, 0u) << rb.first_violation;
}

TEST(VecEnvTest, EpisodesEndWithZeroSumRewardsAcrossAllSeats) {
  Config cfg;
  cfg.ticks = 20;
  VecEnv env(8, 4, {}, cfg, 5, true);
  std::vector<float> obs(8 * 4 * VecEnv::kObsDim), rew(8 * 4);
  std::vector<uint8_t> done(8);
  std::vector<int32_t> act(8 * 4 * 3, 0);
  env.reset(obs.data());
  Rng rng(1);
  int finished = 0;
  for (int t = 0; t < 60; ++t) {
    for (size_t i = 0; i < act.size(); i += 3) {
      act[i] = static_cast<int32_t>(rng.below(8));
      act[i + 1] = static_cast<int32_t>(rng.below(4));
      act[i + 2] = 1 + static_cast<int32_t>(rng.below(30));
    }
    env.step(act.data(), obs.data(), rew.data(), done.data());
    for (int e = 0; e < 8; ++e) {
      if (done[e]) {
        ++finished;
        float s = rew[e * 4] + rew[e * 4 + 1] + rew[e * 4 + 2] + rew[e * 4 + 3];
        EXPECT_NEAR(s, 0.0f, 1e-3f);
      }
    }
    for (float x : obs) ASSERT_TRUE(std::isfinite(x));
  }
  EXPECT_EQ(finished, 8 * 3);
  EXPECT_EQ(env.violations(), 0u);
}
