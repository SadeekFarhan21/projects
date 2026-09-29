#include "figgie/game.hpp"

#include <algorithm>
#include <stdexcept>

namespace figgie {

namespace {
// Error accumulator that allocates nothing unless an invariant fails.
struct Err {
  std::string s;
  Err& operator<<(const char* c) {
    s += c;
    return *this;
  }
  Err& operator<<(int v) {
    s += std::to_string(v);
    return *this;
  }
  std::string str() const { return s; }
};
}  // namespace

Game::Game(Config cfg, uint64_t seed) : cfg_(cfg) {
  if (cfg_.n_players != 4 && cfg_.n_players != 5)
    throw std::invalid_argument("n_players must be 4 or 5");
  if (cfg_.ticks < 1) throw std::invalid_argument("ticks must be >= 1");
  if (cfg_.start_cash < cfg_.ante()) throw std::invalid_argument("start_cash below ante");
  reset(seed);
}

void Game::reset(uint64_t seed) {
  rng_.reseed(seed);
  // Assign {12, 10, 10, 8} to the four suits uniformly (Fisher-Yates).
  Counts sizes{12, 10, 10, 8};
  for (int i = kSuits - 1; i > 0; --i) std::swap(sizes[i], sizes[rng_.below(i + 1)]);
  // Shuffle the physical deck and deal.
  std::array<int, kDeckSize> deck{};
  int k = 0;
  for (int s = 0; s < kSuits; ++s)
    for (int c = 0; c < sizes[s]; ++c) deck[k++] = s;
  for (int i = kDeckSize - 1; i > 0; --i) std::swap(deck[i], deck[rng_.below(i + 1)]);
  std::vector<Counts> hands(cfg_.n_players, Counts{0, 0, 0, 0});
  const int hs = cfg_.hand_size();
  for (int p = 0; p < cfg_.n_players; ++p)
    for (int c = 0; c < hs; ++c) hands[p][deck[p * hs + c]]++;
  reset_with_deal(sizes, hands, rng_.next());
}

void Game::reset_with_deal(const Counts& suit_counts, const std::vector<Counts>& hands, uint64_t seed) {
  Counts sorted = suit_counts;
  std::sort(sorted.begin(), sorted.end());
  if (!(sorted == Counts{8, 10, 10, 12})) throw std::invalid_argument("suit counts must be 12/10/10/8");
  if (static_cast<int>(hands.size()) != cfg_.n_players) throw std::invalid_argument("wrong number of hands");
  Counts sum{0, 0, 0, 0};
  for (const auto& h : hands) {
    if (total(h) != cfg_.hand_size()) throw std::invalid_argument("wrong hand size");
    for (int s = 0; s < kSuits; ++s) {
      if (h[s] < 0) throw std::invalid_argument("negative count");
      sum[s] += h[s];
    }
  }
  if (sum != suit_counts) throw std::invalid_argument("hands do not match suit counts");

  rng_.reseed(seed);
  suit_counts_ = suit_counts;
  int twelve = 0;
  for (int s = 0; s < kSuits; ++s)
    if (suit_counts_[s] == 12) twelve = s;
  goal_ = partner(twelve);
  hands_ = {};
  initial_hands_ = {};
  cash_ = {};
  for (int p = 0; p < cfg_.n_players; ++p) {
    hands_[p] = hands[p];
    initial_hands_[p] = hands[p];
    cash_[p] = cfg_.start_cash - cfg_.ante();
  }
  pot_ = cfg_.pot();
  for (auto& b : books_) b.clear();
  trades_.clear();
  seq_ = 0;
  tick_ = 0;
  done_ = false;
  rejected_ = 0;
  settlement_ = Settlement{};
}

std::array<int, kMaxPlayers> Game::action_order() {
  std::array<int, kMaxPlayers> order{};
  for (int i = 0; i < cfg_.n_players; ++i) order[i] = i;
  for (int i = cfg_.n_players - 1; i > 0; --i) std::swap(order[i], order[rng_.below(i + 1)]);
  return order;
}

Status Game::apply(int player, const Action& a) {
  if (done_ || player < 0 || player >= cfg_.n_players) {
    ++rejected_;
    return Status::kRejected;
  }
  if (a.type != ActType::kPass && a.type != ActType::kCancelAll && (a.suit < 0 || a.suit >= kSuits)) {
    ++rejected_;
    return Status::kRejected;
  }
  switch (a.type) {
    case ActType::kPass:
      return Status::kNoop;
    case ActType::kBid:
      return place(player, a.suit, SuitBook::kBidSide, a.price);
    case ActType::kAsk:
      return place(player, a.suit, SuitBook::kAskSide, a.price);
    case ActType::kBuy:
      return take(player, a.suit, SuitBook::kAskSide);
    case ActType::kSell:
      return take(player, a.suit, SuitBook::kBidSide);
    case ActType::kCancelBid:
      return books_[a.suit].cancel(SuitBook::kBidSide, player) ? Status::kOk : Status::kNoop;
    case ActType::kCancelAsk:
      return books_[a.suit].cancel(SuitBook::kAskSide, player) ? Status::kOk : Status::kNoop;
    case ActType::kCancelAll: {
      bool any = false;
      for (auto& b : books_) {
        any |= b.cancel(SuitBook::kBidSide, player);
        any |= b.cancel(SuitBook::kAskSide, player);
      }
      return any ? Status::kOk : Status::kNoop;
    }
  }
  ++rejected_;
  return Status::kRejected;
}

Status Game::place(int player, int suit, SuitBook::Side side, int price) {
  if (price < 1 || price > cfg_.max_price) {
    ++rejected_;
    return Status::kRejected;
  }
  if (side == SuitBook::kBidSide && price > cash_[player]) {
    ++rejected_;
    return Status::kRejected;
  }
  if (side == SuitBook::kAskSide && hands_[player][suit] < 1) {
    ++rejected_;
    return Status::kRejected;
  }
  SuitBook& b = books_[suit];
  if (b.crosses(side, price)) {
    SuitBook::Side opp = side == SuitBook::kBidSide ? SuitBook::kAskSide : SuitBook::kBidSide;
    const Order best = *b.best(opp);
    if (best.player == player) {  // self-trade prevention: reject the incoming order
      ++rejected_;
      return Status::kRejected;
    }
    b.remove_at(opp, 0);
    if (side == SuitBook::kBidSide)
      execute(suit, player, best.player, best.price);
    else
      execute(suit, best.player, player, best.price);
    return Status::kTraded;
  }
  b.insert(side, player, price, ++seq_);
  return Status::kOk;
}

Status Game::take(int player, int suit, SuitBook::Side resting_side) {
  SuitBook& b = books_[suit];
  auto best = b.best(resting_side);
  if (!best) return Status::kNoop;
  if (best->player == player) {
    ++rejected_;
    return Status::kRejected;
  }
  if (resting_side == SuitBook::kAskSide) {  // we buy
    if (cash_[player] < best->price) {
      ++rejected_;
      return Status::kRejected;
    }
    b.remove_at(resting_side, 0);
    execute(suit, player, best->player, best->price);
  } else {  // we sell
    if (hands_[player][suit] < 1) {
      ++rejected_;
      return Status::kRejected;
    }
    b.remove_at(resting_side, 0);
    execute(suit, best->player, player, best->price);
  }
  return Status::kTraded;
}

void Game::execute(int suit, int buyer, int seller, int price) {
  hands_[seller][suit]--;
  hands_[buyer][suit]++;
  cash_[buyer] -= price;
  cash_[seller] += price;
  trades_.push_back(Trade{tick_, suit, price, buyer, seller});
  if (cfg_.clear_on_trade) {
    for (auto& bk : books_) bk.clear();
  } else {
    reconcile_after_trade(buyer, seller, suit);
  }
}

// Without the clear-on-trade rule we still keep every resting quote backed:
// a bid never exceeds its owner's cash and an offer is never for a card the
// owner no longer holds. Stale quotes are cancelled.
void Game::reconcile_after_trade(int buyer, int seller, int suit) {
  for (auto& bk : books_) {
    int i = bk.find(SuitBook::kBidSide, buyer);
    if (i >= 0 && bk.at(SuitBook::kBidSide, i).price > cash_[buyer]) bk.remove_at(SuitBook::kBidSide, i);
  }
  if (hands_[seller][suit] < 1) books_[suit].cancel(SuitBook::kAskSide, seller);
}

void Game::end_tick() {
  if (done_) return;
  ++tick_;
  if (tick_ >= cfg_.ticks) finalize();
}

void Game::finalize() {
  for (auto& b : books_) b.clear();
  std::vector<Counts> hands(hands_.begin(), hands_.begin() + cfg_.n_players);
  settlement_ = settle(cfg_, goal_, hands);
  pot_ = 0;
  done_ = true;
}

Settlement Game::settle(const Config& cfg, int goal, const std::vector<Counts>& hands) {
  Settlement st;
  const int n = static_cast<int>(hands.size());
  int n_goal = 0, best = 0;
  for (int p = 0; p < n; ++p) {
    st.goal_cards[p] = hands[p][goal];
    n_goal += hands[p][goal];
    best = std::max(best, hands[p][goal]);
  }
  st.remainder = cfg.pot() - 10 * n_goal;
  for (int p = 0; p < n; ++p)
    if (hands[p][goal] == best) st.n_winners++;
  for (int p = 0; p < n; ++p) {
    st.payout_x60[p] = 600LL * hands[p][goal];
    if (hands[p][goal] == best) st.payout_x60[p] += 60LL * st.remainder / st.n_winners;
  }
  return st;
}

double Game::pnl(int p) const {
  double end_cash = cash_[p] + (done_ ? settlement_.payout_x60[p] / 60.0 : 0.0);
  return end_cash - cfg_.start_cash;
}

std::string Game::check_invariants() const {
  Err err;
  const int n = cfg_.n_players;
  Counts sorted = suit_counts_;
  std::sort(sorted.begin(), sorted.end());
  if (!(sorted == Counts{8, 10, 10, 12})) err << "suit counts not 12/10/10/8; ";
  if (suit_counts_[partner(goal_)] != 12) err << "goal is not partner of 12 suit; ";
  if (suit_counts_[goal_] != 8 && suit_counts_[goal_] != 10) err << "goal suit size not 8 or 10; ";
  for (int s = 0; s < kSuits; ++s) {
    int sum = 0, init = 0;
    for (int p = 0; p < n; ++p) {
      if (hands_[p][s] < 0) err << "negative holding p" << p << " s" << s << "; ";
      sum += hands_[p][s];
      init += initial_hands_[p][s];
    }
    if (sum != suit_counts_[s]) err << "cards not conserved in suit " << s << "; ";
    if (init != suit_counts_[s]) err << "initial deal wrong in suit " << s << "; ";
  }
  for (int p = 0; p < n; ++p)
    if (total(initial_hands_[p]) != cfg_.hand_size()) err << "hand size wrong p" << p << "; ";
  for (int p = n; p < kMaxPlayers; ++p)
    if (total(hands_[p]) != 0 || cash_[p] != 0) err << "phantom player state; ";

  int64_t money_x60 = 60LL * pot_;
  for (int p = 0; p < n; ++p) {
    if (cash_[p] < 0) err << "negative cash p" << p << "; ";
    money_x60 += 60LL * cash_[p] + (done_ ? settlement_.payout_x60[p] : 0);
  }
  if (money_x60 != 60LL * n * cfg_.start_cash) err << "money not conserved; ";
  if (!done_ && pot_ != cfg_.pot()) err << "pot changed during play; ";

  for (int s = 0; s < kSuits; ++s) {
    const SuitBook& b = books_[s];
    if (!b.well_formed()) err << "book " << s << " malformed or crossed; ";
    if (done_ && (b.size(SuitBook::kBidSide) || b.size(SuitBook::kAskSide))) err << "quotes after end; ";
    for (int i = 0; i < b.size(SuitBook::kBidSide); ++i) {
      const Order& o = b.at(SuitBook::kBidSide, i);
      if (o.player < 0 || o.player >= n) err << "bad bid owner; ";
      else if (o.price > cash_[o.player]) err << "bid exceeds cash; ";
      if (o.price < 1 || o.price > cfg_.max_price) err << "bid price out of range; ";
    }
    for (int i = 0; i < b.size(SuitBook::kAskSide); ++i) {
      const Order& o = b.at(SuitBook::kAskSide, i);
      if (o.player < 0 || o.player >= n) err << "bad ask owner; ";
      else if (hands_[o.player][s] < 1) err << "offer without card; ";
      if (o.price < 1 || o.price > cfg_.max_price) err << "ask price out of range; ";
    }
  }
  // Replay trades from the initial deal: they must reproduce current holdings and cash.
  std::array<Counts, kMaxPlayers> h = initial_hands_;
  std::array<int, kMaxPlayers> c{};
  for (int p = 0; p < n; ++p) c[p] = cfg_.start_cash - cfg_.ante();
  for (const Trade& t : trades_) {
    if (t.buyer == t.seller) err << "self trade; ";
    if (t.price < 1 || t.price > cfg_.max_price) err << "trade price out of range; ";
    h[t.seller][t.suit]--;
    h[t.buyer][t.suit]++;
    if (h[t.seller][t.suit] < 0) err << "sold a card not held; ";
    c[t.buyer] -= t.price;
    c[t.seller] += t.price;
    if (c[t.buyer] < 0) err << "bought without cash; ";
  }
  for (int p = 0; p < n; ++p)
    if (h[p] != hands_[p] || c[p] != cash_[p]) err << "trade log does not replay; ";
  if (done_) {
    std::vector<Counts> hv(hands_.begin(), hands_.begin() + n);
    Settlement ref = settle(cfg_, goal_, hv);
    int64_t paid = 0;
    int best = 0;
    for (int p = 0; p < n; ++p) {
      paid += settlement_.payout_x60[p];
      best = std::max(best, hands_[p][goal_]);
      if (settlement_.payout_x60[p] != ref.payout_x60[p]) err << "settlement mismatch; ";
    }
    if (paid != 60LL * cfg_.pot()) err << "pot not fully paid; ";
    for (int p = 0; p < n; ++p) {
      int64_t base = 600LL * hands_[p][goal_];
      bool winner = hands_[p][goal_] == best;
      if (!winner && settlement_.payout_x60[p] != base) err << "loser got bonus; ";
      if (winner && settlement_.payout_x60[p] <= base) err << "winner missed bonus; ";
    }
  }
  return err.str();
}

}  // namespace figgie
