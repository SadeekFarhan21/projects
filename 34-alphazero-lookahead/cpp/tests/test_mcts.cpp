#include <gtest/gtest.h>

#include <numeric>

#include "c4/puct.hpp"
#include "c4/pure_mcts.hpp"
#include "c4/solver.hpp"

using namespace c4;

TEST(PureMCTS, TakesImmediateWin) {
  PureMCTS m(1000, 1);
  EXPECT_EQ(m.choose_move(Position::from_moves("121212")), 0);
}

TEST(PureMCTS, BlocksImmediateLoss) {
  PureMCTS m(2000, 2);
  // O to move; X threatens column 1 vertically.
  EXPECT_EQ(m.choose_move(Position::from_moves("12121")), 0);
}

TEST(PuctTree, VisitsSumToSimsWithUniformPolicy) {
  PuctTree t;
  PuctConfig cfg;
  cfg.num_sims = 50;
  cfg.dirichlet_eps = 0.0f;
  std::mt19937_64 rng(0);
  t.reset(Position());
  const float uniform[kWidth] = {1, 1, 1, 1, 1, 1, 1};
  int leaf;
  while ((leaf = t.select_leaf(cfg.num_sims, cfg)) >= 0) t.expand_and_backup(leaf, uniform, 0.0f, cfg, rng);
  EXPECT_EQ(t.sims(), 50);
  EXPECT_EQ(t.node(0).visits, 50u);
  auto pi = t.policy_target();
  EXPECT_NEAR(std::accumulate(pi.begin(), pi.end(), 0.0f), 1.0f, 1e-5);
}

TEST(PuctTree, FindsWinThroughTerminalBackups) {
  // With a uniform, uninformative network the terminal win still dominates.
  PuctTree t;
  PuctConfig cfg;
  cfg.num_sims = 200;
  cfg.dirichlet_eps = 0.0f;
  std::mt19937_64 rng(0);
  t.reset(Position::from_moves("121212"));
  const float uniform[kWidth] = {1, 1, 1, 1, 1, 1, 1};
  int leaf;
  while ((leaf = t.select_leaf(cfg.num_sims, cfg)) >= 0) t.expand_and_backup(leaf, uniform, 0.0f, cfg, rng);
  EXPECT_EQ(t.choose_move(false, rng), 0);
  EXPECT_GT(t.root_value(), 0.5f);
}

TEST(PuctTree, IgnoresIllegalPriorMass) {
  PuctTree t;
  PuctConfig cfg;
  cfg.num_sims = 1;
  cfg.dirichlet_eps = 0.0f;
  std::mt19937_64 rng(0);
  t.reset(Position::from_moves("444444"));
  float policy[kWidth] = {0, 0, 0, 10, 0, 1, 0};  // column 3 is full
  const int leaf = t.select_leaf(cfg.num_sims, cfg);
  t.expand_and_backup(leaf, policy, 0.0f, cfg, rng);
  auto pi = t.policy_target();
  EXPECT_EQ(pi[3], 0.0f);
  EXPECT_FLOAT_EQ(pi[5], 1.0f);
  EXPECT_EQ(t.choose_move(false, rng), 5);
}

namespace {
template <class E>
void run_uniform(E& e) {
  std::vector<float> obs(static_cast<std::size_t>(e.capacity()) * kObsSize);
  std::vector<float> pol(static_cast<std::size_t>(e.capacity()) * kWidth, 1.0f);
  std::vector<float> val(static_cast<std::size_t>(e.capacity()), 0.0f);
  int n;
  while ((n = e.gather(obs.data())) > 0) e.scatter(pol.data(), val.data());
}
}  // namespace

TEST(BatchedGames, SelfPlayProducesConsistentSamples) {
  PuctConfig cfg;
  cfg.num_sims = 16;
  BatchedGames g(8, 20, cfg, 42);
  run_uniform(g);
  EXPECT_TRUE(g.done());
  EXPECT_EQ(g.finished_games(), 20);
  auto games = g.drain_games();
  ASSERT_EQ(games.size(), 20u);
  auto s = g.drain_samples();
  std::size_t total_moves = 0;
  for (const auto& gr : games) {
    total_moves += gr.moves.size();
    Position p = Position::from_moves(gr.moves);  // replays legally
    EXPECT_TRUE(p.is_terminal());
    if (gr.winner >= 0) EXPECT_EQ(gr.winner, 1 - p.side_to_move());
  }
  EXPECT_EQ(s.z.size(), total_moves);
  EXPECT_EQ(s.obs.size(), total_moves * kObsSize);
  for (float z : s.z) EXPECT_TRUE(z == 0.0f || z == 1.0f || z == -1.0f);
}

TEST(BatchedGames, EvaluationAlternatesSidesAndUsesOpenings) {
  PuctConfig cfg;
  cfg.num_sims = 8;
  cfg.dirichlet_eps = 0.0f;
  BatchedGames g(4, 8, cfg, 1, /*opponent_rollouts=*/50, {"4", "44"});
  run_uniform(g);
  auto games = g.drain_games();
  ASSERT_EQ(games.size(), 8u);
  int side0 = 0;
  for (const auto& gr : games) {
    side0 += gr.net_side == 0;
    EXPECT_EQ(gr.moves[0], '4');
  }
  EXPECT_EQ(side0, 4);
  EXPECT_TRUE(g.drain_samples().z.empty());
}

TEST(BatchedAnalysis, ReturnsOnePolicyPerPosition) {
  PuctConfig cfg;
  cfg.num_sims = 30;
  cfg.dirichlet_eps = 0.0f;
  BatchedAnalysis a({Position(), Position::from_moves("121212")}, cfg, 3);
  run_uniform(a);
  EXPECT_TRUE(a.done());
  auto moves = a.best_moves();
  EXPECT_EQ(moves.size(), 2u);
  EXPECT_EQ(moves[1], 0);
}
