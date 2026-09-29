// Limit order books built from ITCH add/execute/cancel/delete events.
//
// Both books expose:
//   void apply(const MdEvent&)        update from one 'A'/'E'/'X'/'D' event
//   Top  top(uint16_t locate)         best bid/ask price and aggregate size
//   uint64_t qty_at(locate, side, px) aggregate size at one price (tests)
//   std::vector<Level> depth(...)      all non-empty levels, best first (tests)
//
// ArrayBook<Table> is the production book: per symbol, a fixed window of price
// levels indexed by (price - base) / tick, an occupancy bitmap so finding the
// next best level after the top empties is a count-leading-zeros scan, and an
// order table (FlatOrderMap by default). No allocation after construction.
//
// MapBook is the textbook version: std::map per side plus std::unordered_map
// for orders. Used as the reference implementation in tests and as the
// baseline in the ablation.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llt/common.hpp"
#include "llt/messages.hpp"
#include "llt/order_table.hpp"

namespace llt {

struct Top {
    uint32_t bid_px{0};
    uint32_t bid_qty{0};  // 0 means no bid
    uint32_t ask_px{0};
    uint32_t ask_qty{0};  // 0 means no ask
    bool operator==(const Top&) const = default;
};

struct DepthLevel {
    uint32_t price;
    uint64_t qty;
    bool operator==(const DepthLevel&) const = default;
};

struct BookStats {
    uint64_t unknown_ref{0};    // E/X/D for an order we never saw
    uint64_t out_of_band{0};    // adds outside the price window
    uint64_t bad_symbol{0};     // locate >= kMaxSymbols
    uint64_t table_full{0};     // order table insert failed
    uint64_t duplicate_ref{0};  // add for a ref already live
};

// ---------------------------------------------------------------------------
template <class Table = FlatOrderMap>
class ArrayBook {
public:
    // levels: number of price ticks tracked per side per symbol. The window is
    // centred on the first add seen for each symbol (see DESIGN.md).
    explicit ArrayBook(uint32_t tick = 100, uint32_t levels = 4096,
                       std::size_t order_capacity = 1u << 21)
        : tick_(tick), levels_(levels), words_((levels + 63) / 64), orders_(order_capacity) {
        for (auto& s : syms_) {
            s.bid.qty.assign(levels_, 0);
            s.bid.count.assign(levels_, 0);
            s.bid.bits.assign(words_, 0);
            s.ask.qty.assign(levels_, 0);
            s.ask.count.assign(levels_, 0);
            s.ask.bits.assign(words_, 0);
        }
    }

    void apply(const MdEvent& e) noexcept {
        switch (e.type) {
            case 'A': add(e); break;
            case 'E':
            case 'X': reduce(e.order_ref, e.qty); break;
            case 'D': reduce(e.order_ref, UINT32_MAX); break;
            default: break;
        }
    }

    Top top(uint16_t locate) const noexcept {
        Top t;
        if (locate >= kMaxSymbols) return t;
        const Sym& s = syms_[locate];
        if (s.bid.best >= 0) {
            t.bid_px = price_of(s, s.bid.best);
            t.bid_qty = s.bid.qty[s.bid.best];
        }
        if (s.ask.best >= 0) {
            t.ask_px = price_of(s, s.ask.best);
            t.ask_qty = s.ask.qty[s.ask.best];
        }
        return t;
    }

    uint64_t qty_at(uint16_t locate, char side, uint32_t px) const noexcept {
        if (locate >= kMaxSymbols) return 0;
        const Sym& s = syms_[locate];
        int idx;
        if (!s.init || !index_of(s, px, idx)) return 0;
        return (side == 'B' ? s.bid : s.ask).qty[idx];
    }

    std::vector<DepthLevel> depth(uint16_t locate, char side) const {
        std::vector<DepthLevel> out;
        if (locate >= kMaxSymbols) return out;
        const Sym& s = syms_[locate];
        const Side& sd = side == 'B' ? s.bid : s.ask;
        for (uint32_t i = 0; i < levels_; ++i)
            if (sd.qty[i]) out.push_back({price_of(s, static_cast<int>(i)), sd.qty[i]});
        if (side == 'B') std::reverse(out.begin(), out.end());
        return out;
    }

    const BookStats& stats() const noexcept { return stats_; }
    std::size_t live_orders() const noexcept { return orders_.size(); }

private:
    struct Side {
        std::vector<uint32_t> qty;    // aggregate shares per level
        std::vector<uint32_t> count;  // resting orders per level
        std::vector<uint64_t> bits;   // bit i set iff qty[i] > 0
        int best{-1};                 // index of best level, -1 if side empty
    };
    struct Sym {
        bool init{false};
        uint32_t base{0};  // price of level 0
        Side bid, ask;
    };

    uint32_t price_of(const Sym& s, int idx) const noexcept {
        return s.base + static_cast<uint32_t>(idx) * tick_;
    }
    bool index_of(const Sym& s, uint32_t px, int& idx) const noexcept {
        if (px < s.base) return false;
        const uint32_t off = px - s.base;
        if (off % tick_ != 0) return false;
        const uint32_t i = off / tick_;
        if (i >= levels_) return false;
        idx = static_cast<int>(i);
        return true;
    }

    void add(const MdEvent& e) noexcept {
        if (e.locate >= kMaxSymbols) {
            ++stats_.bad_symbol;
            return;
        }
        if (e.qty == 0) return;
        Sym& s = syms_[e.locate];
        if (!s.init) {
            // Centre the window on the first price seen. Round to a tick.
            const uint32_t half = (levels_ / 2) * tick_;
            const uint32_t aligned = e.price - (e.price % tick_);
            s.base = aligned > half ? aligned - half : 0;
            s.init = true;
        }
        int idx = 0;
        const bool in_band = index_of(s, e.price, idx);
        if (!orders_.insert(e.order_ref, OrderInfo{e.price, e.qty, e.locate, e.side, in_band})) {
            if (orders_.find(e.order_ref)) ++stats_.duplicate_ref;
            else ++stats_.table_full;
            return;
        }
        if (!in_band) {
            ++stats_.out_of_band;
            return;
        }
        Side& sd = e.side == 'B' ? s.bid : s.ask;
        sd.qty[idx] += e.qty;
        sd.count[idx] += 1;
        sd.bits[idx >> 6] |= (1ull << (idx & 63));
        if (e.side == 'B') {
            if (idx > sd.best) sd.best = idx;
        } else {
            if (sd.best < 0 || idx < sd.best) sd.best = idx;
        }
    }

    // Removes up to `q` shares from an order; deletes it when it reaches zero.
    void reduce(uint64_t ref, uint32_t q) noexcept {
        OrderInfo* o = orders_.find(ref);
        if (!o) {
            ++stats_.unknown_ref;
            return;
        }
        const uint32_t dq = std::min(q, o->qty);
        o->qty -= dq;
        const bool gone = o->qty == 0;
        if (o->in_band) {
            Sym& s = syms_[o->locate];
            const bool is_bid = o->side == 'B';
            Side& sd = is_bid ? s.bid : s.ask;
            int idx = 0;
            index_of(s, o->price, idx);
            sd.qty[idx] -= dq;
            if (gone) sd.count[idx] -= 1;
            if (sd.qty[idx] == 0) {
                sd.bits[idx >> 6] &= ~(1ull << (idx & 63));
                if (idx == sd.best) sd.best = is_bid ? scan_down(sd, idx) : scan_up(sd, idx);
            }
        }
        if (gone) orders_.erase(ref);
    }

    // Highest set bit strictly below idx, or -1.
    int scan_down(const Side& sd, int idx) const noexcept {
        if (idx <= 0) return -1;
        int w = (idx - 1) >> 6;
        const int b = (idx - 1) & 63;
        uint64_t word = sd.bits[w] & (b == 63 ? ~0ull : ((1ull << (b + 1)) - 1));
        while (true) {
            if (word) return (w << 6) + 63 - __builtin_clzll(word);
            if (--w < 0) return -1;
            word = sd.bits[w];
        }
    }
    // Lowest set bit strictly above idx, or -1.
    int scan_up(const Side& sd, int idx) const noexcept {
        int start = idx + 1;
        if (start >= static_cast<int>(levels_)) return -1;
        int w = start >> 6;
        uint64_t word = sd.bits[w] & (~0ull << (start & 63));
        const int nw = static_cast<int>(words_);
        while (true) {
            if (word) return (w << 6) + __builtin_ctzll(word);
            if (++w >= nw) return -1;
            word = sd.bits[w];
        }
    }

    uint32_t tick_;
    uint32_t levels_;
    uint32_t words_;
    std::array<Sym, kMaxSymbols> syms_;
    Table orders_;
    BookStats stats_;
};

// ---------------------------------------------------------------------------
class MapBook {
public:
    explicit MapBook(uint32_t = 100, uint32_t = 4096, std::size_t = 0) {}

    void apply(const MdEvent& e) {
        switch (e.type) {
            case 'A': add(e); break;
            case 'E':
            case 'X': reduce(e.order_ref, e.qty); break;
            case 'D': reduce(e.order_ref, UINT32_MAX); break;
            default: break;
        }
    }

    Top top(uint16_t locate) const {
        Top t;
        if (locate >= kMaxSymbols) return t;
        const Sym& s = syms_[locate];
        if (!s.bids.empty()) {
            t.bid_px = s.bids.begin()->first;
            t.bid_qty = static_cast<uint32_t>(s.bids.begin()->second);
        }
        if (!s.asks.empty()) {
            t.ask_px = s.asks.begin()->first;
            t.ask_qty = static_cast<uint32_t>(s.asks.begin()->second);
        }
        return t;
    }

    uint64_t qty_at(uint16_t locate, char side, uint32_t px) const {
        if (locate >= kMaxSymbols) return 0;
        const Sym& s = syms_[locate];
        if (side == 'B') {
            auto it = s.bids.find(px);
            return it == s.bids.end() ? 0 : it->second;
        }
        auto it = s.asks.find(px);
        return it == s.asks.end() ? 0 : it->second;
    }

    std::vector<DepthLevel> depth(uint16_t locate, char side) const {
        std::vector<DepthLevel> out;
        if (locate >= kMaxSymbols) return out;
        const Sym& s = syms_[locate];
        if (side == 'B')
            for (auto& [p, q] : s.bids) out.push_back({p, q});
        else
            for (auto& [p, q] : s.asks) out.push_back({p, q});
        return out;
    }

    const BookStats& stats() const noexcept { return stats_; }
    std::size_t live_orders() const noexcept { return orders_.size(); }

private:
    struct Sym {
        std::map<uint32_t, uint64_t, std::greater<>> bids;
        std::map<uint32_t, uint64_t> asks;
    };

    void add(const MdEvent& e) {
        if (e.locate >= kMaxSymbols) {
            ++stats_.bad_symbol;
            return;
        }
        if (e.qty == 0) return;
        if (!orders_.emplace(e.order_ref, OrderInfo{e.price, e.qty, e.locate, e.side, true}).second) {
            ++stats_.duplicate_ref;
            return;
        }
        Sym& s = syms_[e.locate];
        if (e.side == 'B') s.bids[e.price] += e.qty;
        else s.asks[e.price] += e.qty;
    }

    void reduce(uint64_t ref, uint32_t q) {
        auto it = orders_.find(ref);
        if (it == orders_.end()) {
            ++stats_.unknown_ref;
            return;
        }
        OrderInfo& o = it->second;
        const uint32_t dq = std::min(q, o.qty);
        o.qty -= dq;
        Sym& s = syms_[o.locate];
        if (o.side == 'B') dec(s.bids, o.price, dq);
        else dec(s.asks, o.price, dq);
        if (o.qty == 0) orders_.erase(it);
    }

    template <class M>
    static void dec(M& m, uint32_t px, uint32_t dq) {
        auto it = m.find(px);
        if (it == m.end()) return;
        it->second -= dq;
        if (it->second == 0) m.erase(it);
    }

    std::array<Sym, kMaxSymbols> syms_;
    std::unordered_map<uint64_t, OrderInfo> orders_;
    BookStats stats_;
};

}  // namespace llt
