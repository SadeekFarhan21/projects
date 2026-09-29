#include "figgie/posterior.hpp"

#include <cmath>
#include <algorithm>

namespace figgie {

namespace {

struct Binom {
  double c[41][41];
  Binom() {
    for (int n = 0; n <= 40; ++n) {
      c[n][0] = 1.0;
      for (int k = 1; k <= 40; ++k) c[n][k] = n == 0 ? 0.0 : c[n - 1][k - 1] + (k <= n - 1 ? c[n - 1][k] : 0.0);
    }
  }
  double operator()(int n, int k) const { return (k < 0 || n < 0 || k > n) ? 0.0 : c[n][k]; }
};
const Binom& binom() {
  static const Binom b;
  return b;
}

std::array<DeckConfig, kNumConfigs> make_configs() {
  std::array<DeckConfig, kNumConfigs> out{};
  int k = 0;
  for (int twelve = 0; twelve < kSuits; ++twelve) {
    for (int eight = 0; eight < kSuits; ++eight) {
      if (eight == twelve) continue;
      Counts sz{10, 10, 10, 10};
      sz[twelve] = 12;
      sz[eight] = 8;
      out[k++] = DeckConfig{sz, partner(twelve)};
    }
  }
  return out;
}

// Enumerate all 4-suit compositions x with x <= cap, x >= lo, |x| = n.
template <class F>
void for_each_composition(const Counts& lo, const Counts& cap, int n, F&& f) {
  Counts x{};
  for (x[0] = lo[0]; x[0] <= cap[0] && x[0] <= n; ++x[0]) {
    int r0 = n - x[0];
    for (x[1] = lo[1]; x[1] <= cap[1] && x[1] <= r0; ++x[1]) {
      int r1 = r0 - x[1];
      for (x[2] = lo[2]; x[2] <= cap[2] && x[2] <= r1; ++x[2]) {
        x[3] = r1 - x[2];
        if (x[3] < lo[3] || x[3] > cap[3]) continue;
        f(x);
      }
    }
  }
}

inline int encode(const Counts& r) { return ((r[0] * 13 + r[1]) * 13 + r[2]) * 13 + r[3]; }

}  // namespace

const std::array<DeckConfig, kNumConfigs>& all_configs() {
  static const auto cfgs = make_configs();
  return cfgs;
}

double hand_likelihood(const Counts& sizes, const Counts& hand) {
  const Binom& C = binom();
  double l = 1.0;
  for (int s = 0; s < kSuits; ++s) l *= C(sizes[s], hand[s]);
  return l;
}

double constraint_probability(const Counts& remaining, const std::vector<Counts>& mins, int hand_size) {
  const int k = static_cast<int>(mins.size());
  if (k == 0) return 1.0;
  bool trivial = true;
  for (const auto& m : mins)
    for (int v : m) trivial &= (v == 0);
  if (trivial) return 1.0;
  for (int s = 0; s < kSuits; ++s) {
    int need = 0;
    for (const auto& m : mins) need += m[s];
    if (need > remaining[s]) return 0.0;
  }
  const Binom& C = binom();
  // two_way(r, a, b): weighted count of splitting r into player a (hand_size
  // cards, at least mins[a]) and player b (the rest, at least mins[b]). The
  // per-suit choices are independent apart from the total, so this is the
  // coefficient of z^hand_size in prod_s sum_{x_s} C(r_s, x_s) z^{x_s}.
  auto two_way = [&](const Counts& r, const Counts& ma, const Counts& mb) -> double {
    double poly[kDeckSize + 1] = {1.0};
    int deg = 0;
    for (int s = 0; s < kSuits; ++s) {
      int lo = ma[s], hi = std::min(r[s] - mb[s], hand_size);
      if (lo > hi) return 0.0;
      double next[kDeckSize + 1] = {0.0};
      for (int d = 0; d <= deg; ++d) {
        if (poly[d] == 0.0) continue;
        for (int x = lo; x <= hi && d + x <= hand_size; ++x) next[d + x] += poly[d] * C(r[s], x);
      }
      deg = std::min(deg + hi, hand_size);
      std::copy(next, next + deg + 1, poly);
    }
    return deg == hand_size ? poly[hand_size] : 0.0;
  };
  // memo[j][encode(r)] = weighted count of deals of r into players j..k-1
  // satisfying their bounds (NaN = not yet computed). Only needed when three
  // or more hands remain, i.e. in 5-player games.
  constexpr int kStates = 13 * 13 * 13 * 13;
  thread_local std::vector<std::vector<double>> memo;
  thread_local std::vector<std::pair<int, int>> touched;
  if (static_cast<int>(memo.size()) < k) memo.resize(k, std::vector<double>(kStates, std::nan("")));
  for (auto [lvl, idx] : touched) memo[lvl][idx] = std::nan("");
  touched.clear();
  struct Rec {
    const std::vector<Counts>& mins;
    int k, hand_size;
    const Binom& C;
    decltype(two_way)& tw;
    double operator()(int j, const Counts& r) const {
      if (j == k - 1) {
        for (int s = 0; s < kSuits; ++s)
          if (r[s] < mins[j][s]) return 0.0;
        return 1.0;  // the last player takes everything left
      }
      if (j == k - 2) return tw(r, mins[j], mins[j + 1]);
      const int key = encode(r);
      const double cached = memo[j][key];
      if (!std::isnan(cached)) return cached;
      double acc = 0.0;
      for_each_composition(mins[j], r, hand_size, [&](const Counts& x) {
        double w = C(r[0], x[0]) * C(r[1], x[1]) * C(r[2], x[2]) * C(r[3], x[3]);
        Counts rest{r[0] - x[0], r[1] - x[1], r[2] - x[2], r[3] - x[3]};
        acc += w * (*this)(j + 1, rest);
      });
      memo[j][key] = acc;
      touched.emplace_back(j, key);
      return acc;
    }
  };
  Rec ways{mins, k, hand_size, C, two_way};
  double num = ways(0, remaining);
  // Unconstrained count by Vandermonde: prod_j C(cards left before j, hand_size).
  double den = 1.0;
  int left = total(remaining);
  for (int j = 0; j < k - 1; ++j) {
    den *= C(left, hand_size);
    left -= hand_size;
  }
  return num / den;
}

ConfigProbs config_posterior(const Counts& my_hand, const std::vector<Counts>& others_min, int hand_size) {
  ConfigProbs post{};
  double z = 0.0;
  const auto& cfgs = all_configs();
  for (int c = 0; c < kNumConfigs; ++c) {
    const Counts& sz = cfgs[c].sizes;
    bool feasible = true;
    Counts rem{};
    for (int s = 0; s < kSuits; ++s) {
      rem[s] = sz[s] - my_hand[s];
      if (rem[s] < 0) feasible = false;
    }
    double p = 0.0;
    if (feasible) {
      p = hand_likelihood(sz, my_hand);
      if (p > 0.0) p *= constraint_probability(rem, others_min, hand_size);
    }
    post[c] = p;
    z += p;
  }
  if (z > 0.0)
    for (auto& p : post) p /= z;
  return post;
}

std::array<double, kSuits> goal_probs(const ConfigProbs& post) {
  std::array<double, kSuits> g{0, 0, 0, 0};
  const auto& cfgs = all_configs();
  for (int c = 0; c < kNumConfigs; ++c) g[cfgs[c].goal] += post[c];
  return g;
}

std::array<double, kSuits> fair_values(const ConfigProbs& post, int pot) {
  std::array<double, kSuits> v{0, 0, 0, 0};
  const auto& cfgs = all_configs();
  for (int c = 0; c < kNumConfigs; ++c) {
    int g = cfgs[c].goal;
    v[g] += post[c] * static_cast<double>(pot) / cfgs[c].sizes[g];
  }
  return v;
}

}  // namespace figgie
