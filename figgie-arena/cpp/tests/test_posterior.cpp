#include <gtest/gtest.h>

#include <cmath>
#include <set>

#include "figgie/game.hpp"
#include "figgie/posterior.hpp"
#include "figgie/rng.hpp"

using namespace figgie;

namespace {
double comb(int n, int k) {
  if (k < 0 || k > n) return 0;
  double r = 1;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}
}  // namespace

TEST(Posterior, TwelveDistinctConfigs) {
  std::set<Counts> seen;
  for (const auto& c : all_configs()) {
    seen.insert(c.sizes);
    EXPECT_EQ(total(c.sizes), 40);
    EXPECT_EQ(c.sizes[partner(c.goal)], 12);
  }
  EXPECT_EQ(seen.size(), 12u);
}

TEST(Posterior, HandOnlyMatchesHypergeometric) {
  Counts h{5, 2, 2, 1};
  auto post = config_posterior(h, {}, 10);
  double z = 0;
  std::array<double, 12> ref{};
  for (int c = 0; c < 12; ++c) {
    const auto& sz = all_configs()[c].sizes;
    ref[c] = comb(sz[0], h[0]) * comb(sz[1], h[1]) * comb(sz[2], h[2]) * comb(sz[3], h[3]);
    z += ref[c];
  }
  for (int c = 0; c < 12; ++c) EXPECT_NEAR(post[c], ref[c] / z, 1e-12);
  auto g = goal_probs(post);
  EXPECT_NEAR(g[0] + g[1] + g[2] + g[3], 1.0, 1e-12);
  EXPECT_GT(g[kClubs], g[kSpades]);  // long spades suggests spades is the 12 suit
}

TEST(Posterior, ConstraintClosedForm) {
  // One constrained player out of three: P(at least 1 of r cards in 10 of 30).
  for (int r = 1; r <= 10; ++r) {
    Counts rem{r, 10, 10, 10 - r};
    double want = 1.0 - comb(30 - r, 10) / comb(30, 10);
    double got = constraint_probability(rem, {Counts{1, 0, 0, 0}, Counts{0, 0, 0, 0}, Counts{0, 0, 0, 0}}, 10);
    EXPECT_NEAR(got, want, 1e-12) << r;
  }
  EXPECT_EQ(constraint_probability(Counts{2, 10, 10, 8}, {Counts{2, 0, 0, 0}, Counts{1, 0, 0, 0}, Counts{}}, 10), 0.0);
  EXPECT_EQ(constraint_probability(Counts{2, 10, 10, 8}, {Counts{}, Counts{}, Counts{}}, 10), 1.0);
}

TEST(Posterior, ConstraintMatchesMonteCarlo) {
  Counts rem{7, 9, 6, 8};
  std::vector<Counts> mins = {Counts{2, 1, 0, 0}, Counts{0, 3, 1, 0}, Counts{1, 0, 0, 2}};
  double exact = constraint_probability(rem, mins, 10);
  Rng rng(5);
  std::vector<int> cards;
  for (int s = 0; s < 4; ++s)
    for (int i = 0; i < rem[s]; ++i) cards.push_back(s);
  const int N = 300000;
  int hit = 0;
  for (int it = 0; it < N; ++it) {
    for (int i = static_cast<int>(cards.size()) - 1; i > 0; --i) std::swap(cards[i], cards[rng.below(i + 1)]);
    bool ok = true;
    for (int j = 0; j < 3 && ok; ++j) {
      Counts h{};
      for (int c = 0; c < 10; ++c) h[cards[j * 10 + c]]++;
      for (int s = 0; s < 4; ++s) ok &= h[s] >= mins[j][s];
    }
    hit += ok;
  }
  double mc = static_cast<double>(hit) / N;
  double se = std::sqrt(exact * (1 - exact) / N);
  EXPECT_NEAR(mc, exact, 5 * se);
}

TEST(Posterior, FiveCardHandSizeWorks) {
  auto post = config_posterior(Counts{3, 2, 2, 1}, {Counts{1, 0, 0, 0}, Counts{}, Counts{0, 2, 0, 0}, Counts{}}, 8);
  double z = 0;
  for (double p : post) z += p;
  EXPECT_NEAR(z, 1.0, 1e-12);
}

TEST(Posterior, FairValuesAveragePot) {
  ConfigProbs certain{};
  certain[0] = 1.0;  // first config
  auto fv = fair_values(certain, 200);
  const auto& c = all_configs()[0];
  EXPECT_NEAR(fv[c.goal], 200.0 / c.sizes[c.goal], 1e-12);
  for (int s = 0; s < 4; ++s)
    if (s != c.goal) EXPECT_EQ(fv[s], 0.0);
}
