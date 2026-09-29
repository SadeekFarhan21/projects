// Core value types shared by the engine, bots and bindings.
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace figgie {

constexpr int kSuits = 4;
constexpr int kDeckSize = 40;
constexpr int kMaxPlayers = 5;

// Suit order: spades, clubs (black), hearts, diamonds (red).
// The same-colour partner of suit s is s ^ 1.
enum Suit : int { kSpades = 0, kClubs = 1, kHearts = 2, kDiamonds = 3 };

inline constexpr int partner(int suit) { return suit ^ 1; }
inline const char* suit_name(int s) {
  static const char* names[kSuits] = {"spades", "clubs", "hearts", "diamonds"};
  return names[s];
}

using Counts = std::array<int, kSuits>;

inline int total(const Counts& c) { return c[0] + c[1] + c[2] + c[3]; }

struct Config {
  int n_players = 4;        // 4 or 5, as in the official rules
  int ticks = 200;          // discrete-time stand-in for the 4 minute clock
  int start_cash = 350;     // chips before the ante
  int max_price = 100;      // quotes must lie in [1, max_price]
  bool clear_on_trade = true;  // official rule: every trade cancels every quote

  int ante() const { return n_players == 4 ? 50 : 40; }
  int pot() const { return ante() * n_players; }  // always 200
  int hand_size() const { return kDeckSize / n_players; }
};

enum class ActType : uint8_t {
  kPass = 0,
  kBid = 1,        // rest or cross a one-lot bid at price
  kAsk = 2,        // rest or cross a one-lot offer at price
  kBuy = 3,        // lift the best offer (price ignored)
  kSell = 4,       // hit the best bid (price ignored)
  kCancelBid = 5,  // cancel own bid in suit
  kCancelAsk = 6,  // cancel own offer in suit
  kCancelAll = 7,  // cancel all own quotes in every suit
};
constexpr int kNumActTypes = 8;

struct Action {
  ActType type = ActType::kPass;
  int suit = 0;
  int price = 0;
};

enum class Status : uint8_t {
  kOk = 0,         // accepted, no trade
  kTraded = 1,     // accepted and produced a trade
  kRejected = 2,   // invalid under the rules (no state change)
  kNoop = 3,       // pass, or nothing to cancel / take
};

struct Trade {
  int tick;
  int suit;
  int price;
  int buyer;
  int seller;
};

}  // namespace figgie
