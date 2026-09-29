#include "mdp/book.hpp"

#include <algorithm>

namespace mdp {

LevelChange SideBook::set(int64_t price, uint32_t size) {
  const int64_t k = key(price);
  // First element whose key is >= k.
  auto it = std::lower_bound(levels_.begin(), levels_.end(), k,
                             [this](const Level& l, int64_t kk) { return key(l.price) < kk; });
  const bool found = it != levels_.end() && it->price == price;
  if (size == 0) {
    if (!found) return LevelChange::DeleteMissing;
    levels_.erase(it);
    return LevelChange::Delete;
  }
  if (found) {
    it->size = size;
    return LevelChange::Update;
  }
  levels_.insert(it, Level{price, size});
  return LevelChange::Insert;
}

Bbo SymbolBook::compute() const {
  Bbo b;
  if (!bids_.empty()) {
    b.bid_price = bids_.best().price;
    b.bid_size = bids_.best().size;
  }
  if (!asks_.empty()) {
    b.ask_price = asks_.best().price;
    b.ask_size = asks_.best().size;
  }
  return b;
}

bool SymbolBook::apply(const PriceLevelUpdate& u, LevelChange* change) {
  ++counters_.updates;
  SideBook& side = u.side == msg::PriceLevelBuy ? bids_ : asks_;
  LevelChange c = side.set(u.price, u.size);
  if (c == LevelChange::DeleteMissing) ++counters_.delete_missing;
  if (change) *change = c;
  if (!(u.flags & kEventProcessingComplete)) {
    in_transition_ = true;
    return false;
  }
  in_transition_ = false;
  ++counters_.events_completed;
  Bbo now = compute();
  if (now.crossed()) ++counters_.crossed_at_complete;
  if (now.locked()) ++counters_.locked_at_complete;
  if (now == consistent_) return false;
  consistent_ = now;
  return true;
}

}  // namespace mdp
