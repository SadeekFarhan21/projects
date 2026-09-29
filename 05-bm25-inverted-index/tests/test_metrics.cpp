#include <gtest/gtest.h>

#include <cmath>

#include "se/metrics.hpp"

using namespace se;

TEST(Metrics, NdcgHandComputed) {
  Judgments rel{{"a", 2}, {"b", 1}, {"c", 1}};
  // ranking: x a y b  -> DCG = 2/log2(3) + 1/log2(5)
  std::vector<std::string> run{"x", "a", "y", "b"};
  double dcg = 2 / std::log2(3.0) + 1 / std::log2(5.0);
  double idcg = 2 / std::log2(2.0) + 1 / std::log2(3.0) + 1 / std::log2(4.0);
  EXPECT_NEAR(ndcg_at_k(run, rel, 10), dcg / idcg, 1e-12);
  // cut-off at 1 sees nothing relevant
  EXPECT_EQ(ndcg_at_k(run, rel, 1), 0.0);
  // perfect ranking
  EXPECT_NEAR(ndcg_at_k({"a", "b", "c"}, rel, 10), 1.0, 1e-12);
}

TEST(Metrics, IdealUsesAllJudgedEvenBeyondK) {
  Judgments rel;
  for (int i = 0; i < 20; ++i) rel["r" + std::to_string(i)] = 1;
  std::vector<std::string> run;
  for (int i = 0; i < 10; ++i) run.push_back("r" + std::to_string(i));
  EXPECT_NEAR(ndcg_at_k(run, rel, 10), 1.0, 1e-12);
  EXPECT_NEAR(recall_at_k(run, rel, 100), 0.5, 1e-12);
}

TEST(Metrics, Mrr) {
  Judgments rel{{"b", 1}};
  EXPECT_DOUBLE_EQ(mrr_at_k({"a", "b"}, rel, 10), 0.5);
  EXPECT_DOUBLE_EQ(mrr_at_k({"a", "b"}, rel, 1), 0.0);
  EXPECT_DOUBLE_EQ(mrr_at_k({}, rel, 10), 0.0);
}

TEST(Metrics, EmptyJudgments) {
  Judgments rel;
  EXPECT_EQ(ndcg_at_k({"a"}, rel, 10), 0.0);
  EXPECT_EQ(recall_at_k({"a"}, rel, 10), 0.0);
}
