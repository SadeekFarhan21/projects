// A tiny one-lot, price-time priority order book for a single suit.
//
// Figgie books are small: every player may hold at most one resting bid and
// one resting offer per suit, so a side never has more than kMaxPlayers
// orders. We therefore store each side as a short sorted array rather than a
// tree of price levels. Sorting is by price (best first) then by arrival
// sequence number (earliest first).
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "figgie/types.hpp"

namespace figgie {

struct Order {
  int player = -1;
  int price = 0;
  uint64_t seq = 0;
};

struct Fill {
  int price;          // resting order's price (the trade price)
  int resting_player; // owner of the resting order that was hit
};

class SuitBook {
 public:
  enum Side { kBidSide = 0, kAskSide = 1 };

  void clear() { n_[0] = n_[1] = 0; }

  int size(Side side) const { return n_[side]; }
  const Order& at(Side side, int i) const { return orders_[side][i]; }

  std::optional<Order> best(Side side) const {
    if (n_[side] == 0) return std::nullopt;
    return orders_[side][0];
  }

  // Index of the player's order on a side, or -1.
  int find(Side side, int player) const {
    for (int i = 0; i < n_[side]; ++i)
      if (orders_[side][i].player == player) return i;
    return -1;
  }

  bool cancel(Side side, int player) {
    int i = find(side, player);
    if (i < 0) return false;
    remove_at(side, i);
    return true;
  }

  void remove_at(Side side, int i) {
    for (int j = i; j + 1 < n_[side]; ++j) orders_[side][j] = orders_[side][j + 1];
    --n_[side];
  }

  // Would an incoming order at this price cross the opposite side?
  bool crosses(Side incoming_side, int price) const {
    Side opp = incoming_side == kBidSide ? kAskSide : kBidSide;
    if (n_[opp] == 0) return false;
    int best = orders_[opp][0].price;
    return incoming_side == kBidSide ? price >= best : price <= best;
  }

  // Insert a resting order. The player's previous order on this side (if any)
  // is replaced, losing its time priority. Caller must ensure it does not cross.
  void insert(Side side, int player, int price, uint64_t seq) {
    cancel(side, player);
    int n = n_[side];
    int pos = n;
    for (int i = 0; i < n; ++i) {
      const Order& o = orders_[side][i];
      bool better = side == kBidSide ? price > o.price : price < o.price;
      if (better) {  // equal price keeps time priority: new goes after
        pos = i;
        break;
      }
    }
    for (int j = n; j > pos; --j) orders_[side][j] = orders_[side][j - 1];
    orders_[side][pos] = Order{player, price, seq};
    ++n_[side];
  }

  // Sortedness and uniqueness checks used by the invariant checker.
  bool well_formed() const {
    for (int side = 0; side < 2; ++side) {
      if (n_[side] < 0 || n_[side] > kMaxPlayers) return false;
      for (int i = 0; i < n_[side]; ++i) {
        for (int j = i + 1; j < n_[side]; ++j)
          if (orders_[side][i].player == orders_[side][j].player) return false;
        if (i + 1 < n_[side]) {
          const Order& a = orders_[side][i];
          const Order& b = orders_[side][i + 1];
          bool ok = side == kBidSide ? (a.price > b.price || (a.price == b.price && a.seq < b.seq))
                                     : (a.price < b.price || (a.price == b.price && a.seq < b.seq));
          if (!ok) return false;
        }
      }
    }
    if (n_[0] > 0 && n_[1] > 0 && orders_[0][0].price >= orders_[1][0].price) return false;
    return true;
  }

 private:
  std::array<std::array<Order, kMaxPlayers + 1>, 2> orders_{};
  std::array<int, 2> n_{0, 0};
};

}  // namespace figgie
