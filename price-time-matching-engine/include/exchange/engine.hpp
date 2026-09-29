// Single-symbol limit order book and matching engine with price-time
// priority. See DESIGN.md for the data layout and the invariants that
// check_invariants() enforces.
#pragma once

#include "exchange/events.hpp"
#include "exchange/types.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#ifdef EXCH_STD_UNORDERED_MAP
#include <unordered_map>
#else
#include "exchange/id_map.hpp"
#endif

namespace exch {

#ifdef EXCH_STD_UNORDERED_MAP
// Drop-in replacement used only by the id-map ablation benchmark.
class IdMap {
public:
    static constexpr std::uint32_t kMissing = 0xFFFFFFFFu;
    explicit IdMap(std::size_t expected = 1024) { m_.reserve(expected); }
    std::size_t size() const { return m_.size(); }
    std::uint32_t find(OrderId k) const {
        auto it = m_.find(k);
        return it == m_.end() ? kMissing : it->second;
    }
    bool insert(OrderId k, std::uint32_t v) { return m_.emplace(k, v).second; }
    bool erase(OrderId k) { return m_.erase(k) != 0; }
    void clear() { m_.clear(); }

private:
    std::unordered_map<OrderId, std::uint32_t> m_;
};
#endif

struct OrderView {
    OrderId id;
    Side side;
    Price price;
    Qty qty;
    bool operator==(const OrderView&) const = default;
};

using Depth = std::vector<std::pair<Price, Qty>>; // best level first

class MatchingEngine {
public:
    explicit MatchingEngine(std::size_t expected_orders = 1 << 16);

    // Apply one input event. Output events are appended to `out` (which is
    // not cleared). The engine assigns the input its sequence number.
    void process(const InputEvent& in, std::vector<OutputEvent>& out);

    // ---- read-only queries (not on the hot path)
    std::optional<Price> best_price(Side s) const;
    Qty level_qty(Side s, Price p) const;
    std::size_t level_count(Side s) const { return levels_[idx(s)].size(); }
    std::size_t order_count() const { return ids_.size(); }
    Depth depth(Side s, std::size_t max_levels = SIZE_MAX) const;
    std::optional<OrderView> find_order(OrderId id) const;
    std::vector<OrderView> orders_in_priority(Side s) const; // best first, FIFO within level
    SeqNo next_seq() const { return seq_; }

    // Walks every structure and aborts with a message if an invariant is
    // broken. O(orders), used by tests and the fuzz driver.
    void check_invariants() const;

private:
    static constexpr std::uint32_t kNil = 0xFFFFFFFFu;

    struct Order {
        OrderId id;
        Price price;
        Qty qty;
        std::uint32_t prev;
        std::uint32_t next;
        Side side;
    };

    struct Level {
        Price price;
        Qty total;
        std::uint32_t count;
        std::uint32_t head;
        std::uint32_t tail;
    };

    struct Touched {
        Side side;
        Price price;
        Qty before;
    };

    static constexpr int idx(Side s) { return static_cast<int>(s); }
    // True when price a is strictly better than b for side s.
    static bool better(Side s, Price a, Price b) { return s == Side::Buy ? a > b : a < b; }
    // True when an aggressor on `side` with limit `limit` can trade at `level_price`.
    static bool crosses(Side side, Price limit, Price level_price) {
        return side == Side::Buy ? level_price <= limit : level_price >= limit;
    }

    void on_new(const InputEvent& in, std::vector<OutputEvent>& out);
    void on_cancel(const InputEvent& in, std::vector<OutputEvent>& out);
    void on_modify(const InputEvent& in, std::vector<OutputEvent>& out);

    // Match an aggressor against the opposite side. Returns remaining qty.
    Qty match(OrderId taker, Side side, bool has_limit, Price limit, Qty qty,
              std::vector<OutputEvent>& out);
    Qty fillable(Side side, bool has_limit, Price limit, Qty want) const;
    void rest(OrderId id, Side side, Price price, Qty qty);
    // Unlinks a resting order, updates its level, frees the slot. Does not emit.
    void remove(std::uint32_t oi);
    // Index of the level with this price, or the insertion point with found=false.
    std::size_t find_level(Side s, Price p, bool& found) const;

    void touch(Side s, Price p, Qty before);
    void flush_book_updates(std::vector<OutputEvent>& out);

    std::uint32_t alloc_order();

    std::vector<Order> pool_;
    std::vector<std::uint32_t> free_;
    // Levels per side, sorted worst price first so the best level is at the
    // back: removing the touch after a sweep is pop_back().
    std::vector<Level> levels_[2];
    IdMap ids_;
    std::vector<Touched> touched_;
    SeqNo seq_{0};
    TradeId next_trade_{1};
};

} // namespace exch
