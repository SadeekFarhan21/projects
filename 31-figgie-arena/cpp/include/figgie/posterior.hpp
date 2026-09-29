// Exact posterior over the 12 deck configurations.
//
// A configuration is an assignment of sizes {12, 10, 10, 8} to the suits; it
// fixes the goal suit. With a uniform prior, the posterior of configuration c
// for a player who was dealt hand h and has seen the public trade tape is
//
//   P(c | h, tape) ∝ P(h | c) · P(others' hands consistent with tape | c, h)
//
// P(h | c) is multivariate hypergeometric. The tape reveals, for each other
// player j and suit s, a lower bound m[j][s] on how many cards of s player j
// was dealt: a player can only sell cards they hold, so their initial holding
// is at least the running maximum of (sold - bought) in that suit. The second
// factor is the fraction of deals of the remaining cards into the other
// hands that satisfy every lower bound, computed exactly by a memoised
// recursion over the remaining-cards vector.
//
// This treats trades as hard constraints only; it does not model *why* a
// player chose to trade (no behavioural likelihood). See DESIGN.md.
#pragma once

#include <array>
#include <vector>

#include "figgie/types.hpp"

namespace figgie {

struct DeckConfig {
  Counts sizes;  // cards per suit
  int goal;      // goal suit
};

constexpr int kNumConfigs = 12;
const std::array<DeckConfig, kNumConfigs>& all_configs();

using ConfigProbs = std::array<double, kNumConfigs>;

// P(hand | config) up to a config-independent constant.
double hand_likelihood(const Counts& sizes, const Counts& hand);

// Fraction of deals of `remaining` cards into hands of `hand_size` (one per
// entry of `mins`) such that hand j has at least mins[j][s] cards of suit s.
double constraint_probability(const Counts& remaining, const std::vector<Counts>& mins, int hand_size);

// Full posterior. `others_min` holds one lower-bound vector per other player.
ConfigProbs config_posterior(const Counts& my_initial_hand, const std::vector<Counts>& others_min, int hand_size);

std::array<double, kSuits> goal_probs(const ConfigProbs& post);

// Expected pot share per goal-suit card: sum_c P(c) · [goal_c == s] · pot / size_c(goal).
// This is the average (not marginal) value of a card of each suit.
std::array<double, kSuits> fair_values(const ConfigProbs& post, int pot);

}  // namespace figgie
