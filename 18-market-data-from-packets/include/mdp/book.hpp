// Per-symbol aggregated price-level book built from DEEP Price Level Updates.
//
// DEEP describes each atomic book transition as zero or more PLUs with the
// event flag OFF followed by exactly one PLU with the flag ON. Levels are
// applied as they arrive, but the book only exposes a new BBO when the
// transition completes; while a symbol is in transition, readers see the BBO
// from before the transaction (spec section "Consuming Price Level Update
// Messages and Updating the IEX BBO").
#pragma once

#include <cstdint>
#include <vector>

#include "mdp/iex_messages.hpp"

namespace mdp {

struct Level {
  int64_t price;
  uint32_t size;
};

struct Bbo {
  int64_t bid_price = 0;
  uint32_t bid_size = 0;
  int64_t ask_price = 0;
  uint32_t ask_size = 0;
  bool operator==(const Bbo&) const = default;
  bool crossed() const { return bid_size && ask_size && bid_price > ask_price; }
  bool locked() const { return bid_size && ask_size && bid_price == ask_price; }
};

enum class LevelChange : uint8_t { Insert, Update, Delete, DeleteMissing, ZeroInsertIgnored };

// One side of a book. Levels are sorted so the best price is at the back of
// the vector, which keeps the common case (changes near the top) cheap.
class SideBook {
 public:
  explicit SideBook(bool is_bid) : is_bid_(is_bid) {}
  LevelChange set(int64_t price, uint32_t size);
  bool empty() const { return levels_.empty(); }
  const Level& best() const { return levels_.back(); }
  size_t depth() const { return levels_.size(); }
  // Level i counted from the best (0 = best).
  const Level& level(size_t i) const { return levels_[levels_.size() - 1 - i]; }
  void clear() { levels_.clear(); }

 private:
  int64_t key(int64_t price) const { return is_bid_ ? price : -price; }
  bool is_bid_;
  std::vector<Level> levels_;  // ascending by key(price); best at back
};

struct BookCounters {
  uint64_t updates = 0;
  uint64_t events_completed = 0;
  uint64_t delete_missing = 0;
  uint64_t crossed_at_complete = 0;
  uint64_t locked_at_complete = 0;
};

class SymbolBook {
 public:
  // Applies one PLU. Returns true when this PLU completed an event and the
  // consistent BBO changed.
  bool apply(const PriceLevelUpdate& u, LevelChange* change = nullptr);
  bool in_transition() const { return in_transition_; }
  // BBO as of the last completed event. Never reflects a half-applied event.
  const Bbo& bbo() const { return consistent_; }
  const SideBook& bids() const { return bids_; }
  const SideBook& asks() const { return asks_; }
  const BookCounters& counters() const { return counters_; }

 private:
  Bbo compute() const;
  SideBook bids_{true};
  SideBook asks_{false};
  bool in_transition_ = false;
  Bbo consistent_;
  BookCounters counters_;
};

// Maps 8-byte symbols to dense ids with an open-addressing table.
class SymbolTable {
 public:
  SymbolTable() { resize(1 << 14); }
  uint32_t get_or_add(Symbol s) {
    size_t mask = keys_.size() - 1;
    size_t i = hash(s) & mask;
    for (;;) {
      if (keys_[i] == s) return ids_[i];
      if (keys_[i] == 0) break;
      i = (i + 1) & mask;
    }
    if ((count_ + 1) * 2 > keys_.size()) {
      resize(keys_.size() * 2);
      return get_or_add(s);
    }
    keys_[i] = s;
    ids_[i] = static_cast<uint32_t>(symbols_.size());
    symbols_.push_back(s);
    ++count_;
    return ids_[i];
  }
  size_t size() const { return symbols_.size(); }
  Symbol symbol(uint32_t id) const { return symbols_[id]; }

 private:
  static size_t hash(Symbol s) { return static_cast<size_t>((s * 0x9E3779B97F4A7C15ull) >> 17); }
  void resize(size_t n) {
    std::vector<Symbol> old = symbols_;
    keys_.assign(n, 0);
    ids_.assign(n, 0);
    symbols_.clear();
    count_ = 0;
    for (Symbol s : old) get_or_add(s);
  }
  std::vector<Symbol> keys_;
  std::vector<uint32_t> ids_;
  std::vector<Symbol> symbols_;
  size_t count_ = 0;
};

class BookManager {
 public:
  SymbolBook& book(Symbol s) {
    uint32_t id = table_.get_or_add(s);
    if (id >= books_.size()) books_.resize(id + 1);
    return books_[id];
  }
  size_t size() const { return books_.size(); }
  const SymbolTable& symbols() const { return table_; }
  const SymbolBook& book_by_id(uint32_t id) const { return books_[id]; }

 private:
  SymbolTable table_;
  std::vector<SymbolBook> books_;
};

}  // namespace mdp
