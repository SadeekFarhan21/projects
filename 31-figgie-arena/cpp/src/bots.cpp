#include "figgie/bots.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace figgie {

// ---------------------------------------------------------------- TapeTracker

void TapeTracker::reset(int n_players, int seat) {
  n_ = n_players;
  seat_ = seat;
  seen_ = 0;
  net_ = {};
  min_init_ = {};
}

bool TapeTracker::update(const std::vector<Trade>& trades) {
  bool changed = false;
  for (; seen_ < trades.size(); ++seen_) {
    const Trade& t = trades[seen_];
    net_[t.buyer][t.suit]++;
    net_[t.seller][t.suit]--;
    int need = -net_[t.seller][t.suit];
    if (need > min_init_[t.seller][t.suit]) {
      min_init_[t.seller][t.suit] = need;
      if (t.seller != seat_) changed = true;
    }
  }
  return changed;
}

std::vector<Counts> TapeTracker::others_min() const {
  std::vector<Counts> out;
  for (int p = 0; p < n_; ++p)
    if (p != seat_) out.push_back(min_init_[p]);
  return out;
}

// ------------------------------------------------------------- shared quoting

namespace {

struct QuoteParams {
  double edge;       // half-spread around fair value for resting quotes
  bool quote;        // post resting quotes at all
  bool take;         // cross the spread when mispriced
  double take_edge;  // required edge to take
};

// One action per tick: first the most profitable take (if allowed), then the
// first quote that differs from what we want, scanning suits round-robin.
Action fair_value_action(const View& v, const std::array<double, kSuits>& fv, const QuoteParams& qp, int& rr) {
  const int me = v.seat();
  const int maxp = v.config().max_price;
  if (qp.take) {
    double best_gain = 0.0;
    Action best{};
    for (int s = 0; s < kSuits; ++s) {
      const SuitBook& b = v.book(s);
      if (auto a = b.best(SuitBook::kAskSide); a && a->player != me && a->price <= v.my_cash()) {
        double gain = fv[s] - a->price;
        if (gain >= qp.take_edge && gain > best_gain) {
          best_gain = gain;
          best = Action{ActType::kBuy, s, a->price};
        }
      }
      if (auto bd = b.best(SuitBook::kBidSide); bd && bd->player != me && v.my_hand(s) > 0) {
        double gain = bd->price - fv[s];
        if (gain >= qp.take_edge && gain > best_gain) {
          best_gain = gain;
          best = Action{ActType::kSell, s, bd->price};
        }
      }
    }
    if (best.type != ActType::kPass) return best;
  }
  if (!qp.quote) return Action{};
  for (int k = 0; k < kSuits; ++k) {
    int s = (rr + k) % kSuits;
    const SuitBook& b = v.book(s);
    // Desired bid: never cross someone else's offer while quoting passively.
    int want_bid = static_cast<int>(std::floor(fv[s] - qp.edge));
    want_bid = std::min({want_bid, v.my_cash(), maxp});
    if (auto a = b.best(SuitBook::kAskSide)) want_bid = std::min(want_bid, a->price - 1);
    int want_ask = static_cast<int>(std::ceil(fv[s] + qp.edge));
    want_ask = std::max(want_ask, 1);
    if (auto bd = b.best(SuitBook::kBidSide)) want_ask = std::max(want_ask, bd->price + 1);
    bool can_ask = v.my_hand(s) > 0 && want_ask <= maxp;
    bool can_bid = want_bid >= 1;

    int bi = b.find(SuitBook::kBidSide, me);
    int ai = b.find(SuitBook::kAskSide, me);
    int cur_bid = bi >= 0 ? b.at(SuitBook::kBidSide, bi).price : -1;
    int cur_ask = ai >= 0 ? b.at(SuitBook::kAskSide, ai).price : -1;
    if (!can_bid && cur_bid > 0) { rr = s; return Action{ActType::kCancelBid, s, 0}; }
    if (can_bid && cur_bid != want_bid) { rr = s; return Action{ActType::kBid, s, want_bid}; }
    if (!can_ask && cur_ask > 0) { rr = s; return Action{ActType::kCancelAsk, s, 0}; }
    if (can_ask && cur_ask != want_ask) { rr = (s + 1) % kSuits; return Action{ActType::kAsk, s, want_ask}; }
  }
  return Action{};
}

std::array<double, kSuits> uniform_belief() { return {0.25, 0.25, 0.25, 0.25}; }

// ------------------------------------------------------------------- bots

class RandomBot final : public Bot {
 public:
  std::string name() const override { return "random"; }
  void reset(const View&, uint64_t seed) override { rng_.reseed(seed); }
  Action act(const View& v) override {
    if (rng_.uniform() < 0.4) return Action{};
    int s = static_cast<int>(rng_.below(kSuits));
    int price = 1 + static_cast<int>(rng_.below(30));
    switch (rng_.below(6)) {
      case 0: return Action{ActType::kBid, s, std::min(price, std::max(1, v.my_cash()))};
      case 1: {
        // Offer a suit we actually hold, if any.
        std::vector<int> held;
        for (int t = 0; t < kSuits; ++t)
          if (v.my_hand(t) > 0) held.push_back(t);
        if (held.empty()) return Action{};
        return Action{ActType::kAsk, held[rng_.below(static_cast<uint32_t>(held.size()))], price};
      }
      case 2: return Action{ActType::kBuy, s, 0};
      case 3: return Action{ActType::kSell, s, 0};
      case 4: return Action{ActType::kCancelBid, s, 0};
      default: return Action{ActType::kCancelAsk, s, 0};
    }
  }
  std::array<double, kSuits> goal_belief(const View&) override { return uniform_belief(); }

 private:
  Rng rng_;
};

// Quotes a fixed half-spread around the fair value implied by its own hand
// only (exact hand-likelihood posterior); never takes and ignores the tape.
class PassiveBot final : public Bot {
 public:
  std::string name() const override { return "passive"; }
  void reset(const View& v, uint64_t) override {
    post_ = config_posterior(v.my_initial_hand(), {}, v.config().hand_size());
    fv_ = fair_values(post_, v.config().pot());
    rr_ = 0;
  }
  Action act(const View& v) override {
    return fair_value_action(v, fv_, QuoteParams{4.0, true, false, 0.0}, rr_);
  }
  std::array<double, kSuits> goal_belief(const View&) override { return goal_probs(post_); }

 private:
  ConfigProbs post_{};
  std::array<double, kSuits> fv_{};
  int rr_ = 0;
};

// Beginner heuristic: "my longest suit is probably the 12-card suit", so the
// goal is its partner. Values that suit at pot / 10 and aggressively lifts
// offers below that value and hits bids above it. Never quotes.
class TakerBot final : public Bot {
 public:
  std::string name() const override { return "taker"; }
  void reset(const View& v, uint64_t) override {
    const Counts& h = v.my_initial_hand();
    int mx = *std::max_element(h.begin(), h.end());
    belief_ = {0, 0, 0, 0};
    int k = 0;
    for (int s = 0; s < kSuits; ++s)
      if (h[s] == mx) ++k;
    for (int s = 0; s < kSuits; ++s)
      if (h[s] == mx) belief_[partner(s)] += 1.0 / k;
    for (int s = 0; s < kSuits; ++s) fv_[s] = belief_[s] * v.config().pot() / 10.0;
  }
  Action act(const View& v) override {
    int rr = 0;
    return fair_value_action(v, fv_, QuoteParams{0.0, false, true, 0.5}, rr);
  }
  std::array<double, kSuits> goal_belief(const View&) override { return belief_; }

 private:
  std::array<double, kSuits> belief_{};
  std::array<double, kSuits> fv_{};
};

// Exact posterior from own hand plus the lower bounds revealed by the tape,
// quotes around the implied fair value and takes clear mispricings.
class BayesBot final : public Bot {
 public:
  std::string name() const override { return "bayes"; }
  void reset(const View& v, uint64_t) override {
    tape_.reset(v.n_players(), v.seat());
    recompute(v);
    rr_ = 0;
  }
  Action act(const View& v) override {
    if (tape_.update(v.trades())) recompute(v);
    return fair_value_action(v, fv_, QuoteParams{3.0, true, true, 1.0}, rr_);
  }
  std::array<double, kSuits> goal_belief(const View& v) override {
    if (tape_.update(v.trades())) recompute(v);
    return goal_probs(post_);
  }

 private:
  void recompute(const View& v) {
    post_ = config_posterior(v.my_initial_hand(), tape_.others_min(), v.config().hand_size());
    fv_ = fair_values(post_, v.config().pot());
  }
  TapeTracker tape_;
  ConfigProbs post_{};
  std::array<double, kSuits> fv_{};
  int rr_ = 0;
};

}  // namespace

std::vector<std::string> bot_names() { return {"random", "passive", "taker", "bayes"}; }

std::unique_ptr<Bot> make_bot(const std::string& name) {
  if (name == "random") return std::make_unique<RandomBot>();
  if (name == "passive") return std::make_unique<PassiveBot>();
  if (name == "taker") return std::make_unique<TakerBot>();
  if (name == "bayes") return std::make_unique<BayesBot>();
  throw std::invalid_argument("unknown bot: " + name);
}

}  // namespace figgie
