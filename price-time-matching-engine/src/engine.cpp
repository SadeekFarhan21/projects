#include "exchange/engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace exch {

namespace {

[[noreturn]] void invariant_failed(const char* what) {
    std::fprintf(stderr, "engine invariant violated: %s\n", what);
    std::abort();
}

#define EXCH_CHECK(cond) \
    do {                 \
        if (!(cond)) invariant_failed(#cond); \
    } while (0)

OutputEvent make(OutputKind k, SeqNo seq, OrderId id) {
    OutputEvent e;
    e.kind = k;
    e.seq = seq;
    e.id = id;
    return e;
}

} // namespace

MatchingEngine::MatchingEngine(std::size_t expected_orders) : ids_(expected_orders) {
    pool_.reserve(expected_orders);
    free_.reserve(expected_orders);
    levels_[0].reserve(1024);
    levels_[1].reserve(1024);
    touched_.reserve(64);
}

void MatchingEngine::process(const InputEvent& in, std::vector<OutputEvent>& out) {
    switch (in.kind) {
    case InputKind::New: on_new(in, out); break;
    case InputKind::Cancel: on_cancel(in, out); break;
    case InputKind::Modify: on_modify(in, out); break;
    }
    flush_book_updates(out);
    ++seq_;
}

// ------------------------------------------------------------------ inputs

void MatchingEngine::on_new(const InputEvent& in, std::vector<OutputEvent>& out) {
    auto reject = [&](RejectReason r) {
        OutputEvent e = make(OutputKind::Rejected, seq_, in.id);
        e.reject = r;
        out.push_back(e);
    };
    const bool is_market = in.type == OrderType::Market;
    if (in.id == 0) return reject(RejectReason::InvalidId);
    if (ids_.find(in.id) != IdMap::kMissing) return reject(RejectReason::DuplicateId);
    if (in.qty <= 0 || in.qty > kMaxQty) return reject(RejectReason::InvalidQty);
    if (!is_market && (in.price <= 0 || in.price > kMaxPrice)) return reject(RejectReason::InvalidPrice);

    const Side opp = opposite(in.side);
    if (in.type == OrderType::PostOnly) {
        const auto& ol = levels_[idx(opp)];
        if (!ol.empty() && crosses(in.side, in.price, ol.back().price))
            return reject(RejectReason::PostOnlyWouldCross);
    }

    const Price price = is_market ? 0 : in.price;
    OutputEvent acc = make(OutputKind::Accepted, seq_, in.id);
    acc.side = in.side;
    acc.price = price;
    acc.qty = in.qty;
    out.push_back(acc);

    auto cancel_rest = [&](Qty q, CancelReason why) {
        OutputEvent e = make(OutputKind::Canceled, seq_, in.id);
        e.side = in.side;
        e.price = price;
        e.qty = q;
        e.cancel = why;
        out.push_back(e);
    };

    if (in.type == OrderType::FOK && fillable(in.side, true, price, in.qty) < in.qty)
        return cancel_rest(in.qty, CancelReason::FokUnfilled);

    const Qty left = match(in.id, in.side, !is_market, price, in.qty, out);
    if (left == 0) return;
    switch (in.type) {
    case OrderType::Limit:
    case OrderType::PostOnly: rest(in.id, in.side, price, left); break;
    case OrderType::Market:
    case OrderType::IOC:
    case OrderType::FOK: cancel_rest(left, CancelReason::Unfilled); break;
    }
}

void MatchingEngine::on_cancel(const InputEvent& in, std::vector<OutputEvent>& out) {
    const std::uint32_t oi = in.id == 0 ? IdMap::kMissing : ids_.find(in.id);
    if (oi == IdMap::kMissing) {
        OutputEvent e = make(OutputKind::Rejected, seq_, in.id);
        e.reject = RejectReason::UnknownId;
        out.push_back(e);
        return;
    }
    const Order o = pool_[oi];
    remove(oi);
    OutputEvent e = make(OutputKind::Canceled, seq_, o.id);
    e.side = o.side;
    e.price = o.price;
    e.qty = o.qty;
    e.cancel = CancelReason::User;
    out.push_back(e);
}

void MatchingEngine::on_modify(const InputEvent& in, std::vector<OutputEvent>& out) {
    auto reject = [&](RejectReason r) {
        OutputEvent e = make(OutputKind::Rejected, seq_, in.id);
        e.reject = r;
        out.push_back(e);
    };
    const std::uint32_t oi = in.id == 0 ? IdMap::kMissing : ids_.find(in.id);
    if (oi == IdMap::kMissing) return reject(RejectReason::UnknownId);
    if (in.qty <= 0 || in.qty > kMaxQty) return reject(RejectReason::InvalidQty);
    if (in.price <= 0 || in.price > kMaxPrice) return reject(RejectReason::InvalidPrice);

    Order& o = pool_[oi];
    const Side side = o.side;
    OutputEvent m = make(OutputKind::Modified, seq_, o.id);
    m.side = side;
    m.price = in.price;
    m.qty = in.qty;

    if (in.price == o.price && in.qty <= o.qty) {
        // Same price, size not increased: amend in place, keep queue position.
        bool found = false;
        const std::size_t li = find_level(side, o.price, found);
        Level& L = levels_[idx(side)][li];
        touch(side, L.price, L.total);
        L.total -= o.qty - in.qty;
        o.qty = in.qty;
        out.push_back(m);
        return;
    }

    // Price change or size increase: loses priority. The order leaves the
    // book and re-enters as a fresh limit order under the same id, which
    // means it can trade if the new price crosses.
    const OrderId id = o.id;
    remove(oi);
    out.push_back(m);
    const Qty left = match(id, side, true, in.price, in.qty, out);
    if (left > 0) rest(id, side, in.price, left);
}

// ---------------------------------------------------------------- matching

Qty MatchingEngine::match(OrderId taker, Side side, bool has_limit, Price limit, Qty qty,
                          std::vector<OutputEvent>& out) {
    const Side opp = opposite(side);
    auto& book = levels_[idx(opp)];
    while (qty > 0 && !book.empty()) {
        Level& L = book.back();
        if (has_limit && !crosses(side, limit, L.price)) break;
        touch(opp, L.price, L.total);
        while (qty > 0 && L.head != kNil) {
            const std::uint32_t mi = L.head;
            Order& m = pool_[mi];
            const Qty fill = std::min(qty, m.qty);
            OutputEvent t;
            t.kind = OutputKind::Trade;
            t.seq = seq_;
            t.id = taker;
            t.maker_id = m.id;
            t.trade_id = next_trade_++;
            t.side = side;
            t.price = L.price;
            t.qty = fill;
            out.push_back(t);
            m.qty -= fill;
            L.total -= fill;
            qty -= fill;
            if (m.qty == 0) {
                // Maker fully filled: pop it off the front of the queue.
                L.head = m.next;
                if (L.head != kNil) pool_[L.head].prev = kNil;
                else L.tail = kNil;
                --L.count;
                ids_.erase(m.id);
                free_.push_back(mi);
            }
        }
        if (L.head == kNil) book.pop_back();
    }
    return qty;
}

Qty MatchingEngine::fillable(Side side, bool has_limit, Price limit, Qty want) const {
    const auto& book = levels_[idx(opposite(side))];
    Qty avail = 0;
    for (auto it = book.rbegin(); it != book.rend() && avail < want; ++it) {
        if (has_limit && !crosses(side, limit, it->price)) break;
        avail += it->total;
    }
    return avail;
}

// ------------------------------------------------------------ book editing

std::size_t MatchingEngine::find_level(Side s, Price p, bool& found) const {
    const auto& book = levels_[idx(s)];
    // Sorted worst first. Most activity is near the touch (the back), so
    // scan a few levels from the back before falling back to binary search.
    std::size_t n = book.size();
    for (std::size_t k = 0; k < 8 && k < n; ++k) {
        const std::size_t i = n - 1 - k;
        if (book[i].price == p) { found = true; return i; }
        if (better(s, p, book[i].price)) { found = false; return i + 1; }
    }
    auto it = std::lower_bound(book.begin(), book.end(), p,
                               [s](const Level& l, Price v) { return better(s, v, l.price); });
    found = it != book.end() && it->price == p;
    return static_cast<std::size_t>(it - book.begin());
}

std::uint32_t MatchingEngine::alloc_order() {
    if (!free_.empty()) {
        const std::uint32_t i = free_.back();
        free_.pop_back();
        return i;
    }
    pool_.push_back(Order{});
    return static_cast<std::uint32_t>(pool_.size() - 1);
}

void MatchingEngine::rest(OrderId id, Side side, Price price, Qty qty) {
    const std::uint32_t oi = alloc_order();
    auto& book = levels_[idx(side)];
    bool found = false;
    const std::size_t li = find_level(side, price, found);
    if (!found) {
        touch(side, price, 0);
        book.insert(book.begin() + static_cast<std::ptrdiff_t>(li), Level{price, 0, 0, kNil, kNil});
    } else {
        touch(side, price, book[li].total);
    }
    Level& L = book[li];
    pool_[oi] = Order{id, price, qty, L.tail, kNil, side};
    if (L.tail != kNil) pool_[L.tail].next = oi;
    else L.head = oi;
    L.tail = oi;
    L.total += qty;
    ++L.count;
    ids_.insert(id, oi);
}

void MatchingEngine::remove(std::uint32_t oi) {
    Order& o = pool_[oi];
    auto& book = levels_[idx(o.side)];
    bool found = false;
    const std::size_t li = find_level(o.side, o.price, found);
    Level& L = book[li];
    touch(o.side, L.price, L.total);
    if (o.prev != kNil) pool_[o.prev].next = o.next;
    else L.head = o.next;
    if (o.next != kNil) pool_[o.next].prev = o.prev;
    else L.tail = o.prev;
    L.total -= o.qty;
    --L.count;
    if (L.count == 0) book.erase(book.begin() + static_cast<std::ptrdiff_t>(li));
    ids_.erase(o.id);
    free_.push_back(oi);
}

// ------------------------------------------------------------- market data

void MatchingEngine::touch(Side s, Price p, Qty before) {
    for (const Touched& t : touched_)
        if (t.side == s && t.price == p) return; // keep the first "before"
    touched_.push_back(Touched{s, p, before});
}

void MatchingEngine::flush_book_updates(std::vector<OutputEvent>& out) {
    if (touched_.empty()) return;
    // Canonical order so the feed is deterministic: bids then asks, price ascending.
    std::sort(touched_.begin(), touched_.end(), [](const Touched& a, const Touched& b) {
        if (a.side != b.side) return a.side < b.side;
        return a.price < b.price;
    });
    for (const Touched& t : touched_) {
        const Qty now = level_qty(t.side, t.price);
        if (now == t.before) continue; // net no-op, publish nothing
        OutputEvent e;
        e.kind = OutputKind::BookUpdate;
        e.seq = seq_;
        e.side = t.side;
        e.price = t.price;
        e.qty = now;
        out.push_back(e);
    }
    touched_.clear();
}

// ----------------------------------------------------------------- queries

std::optional<Price> MatchingEngine::best_price(Side s) const {
    const auto& book = levels_[idx(s)];
    if (book.empty()) return std::nullopt;
    return book.back().price;
}

Qty MatchingEngine::level_qty(Side s, Price p) const {
    bool found = false;
    const std::size_t li = find_level(s, p, found);
    return found ? levels_[idx(s)][li].total : 0;
}

Depth MatchingEngine::depth(Side s, std::size_t max_levels) const {
    Depth d;
    const auto& book = levels_[idx(s)];
    for (auto it = book.rbegin(); it != book.rend() && d.size() < max_levels; ++it)
        d.emplace_back(it->price, it->total);
    return d;
}

std::optional<OrderView> MatchingEngine::find_order(OrderId id) const {
    const std::uint32_t oi = ids_.find(id);
    if (oi == IdMap::kMissing) return std::nullopt;
    const Order& o = pool_[oi];
    return OrderView{o.id, o.side, o.price, o.qty};
}

std::vector<OrderView> MatchingEngine::orders_in_priority(Side s) const {
    std::vector<OrderView> v;
    const auto& book = levels_[idx(s)];
    for (auto it = book.rbegin(); it != book.rend(); ++it)
        for (std::uint32_t i = it->head; i != kNil; i = pool_[i].next)
            v.push_back(OrderView{pool_[i].id, s, pool_[i].price, pool_[i].qty});
    return v;
}

void MatchingEngine::check_invariants() const {
    std::size_t total_orders = 0;
    for (int si = 0; si < 2; ++si) {
        const Side s = static_cast<Side>(si);
        const auto& book = levels_[si];
        for (std::size_t li = 0; li < book.size(); ++li) {
            const Level& L = book[li];
            if (li > 0) EXCH_CHECK(better(s, L.price, book[li - 1].price)); // strictly sorted
            EXCH_CHECK(L.count > 0 && L.head != kNil && L.tail != kNil);
            Qty sum = 0;
            std::uint32_t n = 0;
            std::uint32_t prev = kNil;
            for (std::uint32_t i = L.head; i != kNil; i = pool_[i].next) {
                const Order& o = pool_[i];
                EXCH_CHECK(o.prev == prev);
                EXCH_CHECK(o.side == s && o.price == L.price && o.qty > 0);
                EXCH_CHECK(ids_.find(o.id) == i);
                sum += o.qty;
                ++n;
                prev = i;
                EXCH_CHECK(n <= pool_.size()); // no cycles
            }
            EXCH_CHECK(prev == L.tail);
            EXCH_CHECK(sum == L.total && n == L.count);
            total_orders += n;
        }
    }
    EXCH_CHECK(total_orders == ids_.size());
    EXCH_CHECK(total_orders + free_.size() == pool_.size());
    const auto bb = best_price(Side::Buy);
    const auto ba = best_price(Side::Sell);
    if (bb && ba) EXCH_CHECK(*bb < *ba); // never crossed after an input completes
}

} // namespace exch
