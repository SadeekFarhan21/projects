#include "exchange/reference.hpp"

#include <algorithm>

namespace exch {

namespace {
OutputEvent ev(OutputKind k, SeqNo seq, OrderId id) {
    OutputEvent e;
    e.kind = k;
    e.seq = seq;
    e.id = id;
    return e;
}
bool can_trade(Side taker, Price limit, Price maker_price) {
    return taker == Side::Buy ? maker_price <= limit : maker_price >= limit;
}
} // namespace

void ReferenceMatcher::process(const InputEvent& in, std::vector<OutputEvent>& out) {
    const Snapshot before = snapshot();
    switch (in.kind) {
    case InputKind::New: new_order(in, out); break;
    case InputKind::Cancel: cancel(in, out); break;
    case InputKind::Modify: modify(in, out); break;
    }
    // L2 by brute force: every (side, price) whose aggregate differs.
    const Snapshot after = snapshot();
    Snapshot keys = before;
    for (const auto& kv : after) keys[kv.first] = 0;
    for (const auto& kv : keys) {
        auto b = before.find(kv.first);
        auto a = after.find(kv.first);
        const Qty qb = b == before.end() ? 0 : b->second;
        const Qty qa = a == after.end() ? 0 : a->second;
        if (qa == qb) continue;
        OutputEvent e = ev(OutputKind::BookUpdate, seq_, 0);
        e.side = static_cast<Side>(kv.first.first);
        e.price = kv.first.second;
        e.qty = qa;
        out.push_back(e);
    }
    ++seq_;
}

ReferenceMatcher::Snapshot ReferenceMatcher::snapshot() const {
    Snapshot s;
    for (const Resting& o : orders_) s[{static_cast<int>(o.side), o.price}] += o.qty;
    return s;
}

int ReferenceMatcher::find(OrderId id) const {
    for (std::size_t i = 0; i < orders_.size(); ++i)
        if (orders_[i].id == id) return static_cast<int>(i);
    return -1;
}

int ReferenceMatcher::best_opposite(Side taker_side) const {
    const Side want = opposite(taker_side);
    int best = -1;
    for (std::size_t i = 0; i < orders_.size(); ++i) {
        const Resting& o = orders_[i];
        if (o.side != want) continue;
        if (best < 0) { best = static_cast<int>(i); continue; }
        const Resting& b = orders_[static_cast<std::size_t>(best)];
        const bool better_price = want == Side::Buy ? o.price > b.price : o.price < b.price;
        if (better_price || (o.price == b.price && o.time < b.time)) best = static_cast<int>(i);
    }
    return best;
}

void ReferenceMatcher::do_match(OrderId taker, Side side, bool has_limit, Price limit, Qty& qty,
                                std::vector<OutputEvent>& out) {
    while (qty > 0) {
        const int bi = best_opposite(side);
        if (bi < 0) break;
        Resting& m = orders_[static_cast<std::size_t>(bi)];
        if (has_limit && !can_trade(side, limit, m.price)) break;
        const Qty f = std::min(qty, m.qty);
        OutputEvent t = ev(OutputKind::Trade, seq_, taker);
        t.maker_id = m.id;
        t.trade_id = next_trade_++;
        t.side = side;
        t.price = m.price;
        t.qty = f;
        out.push_back(t);
        qty -= f;
        m.qty -= f;
        if (m.qty == 0) orders_.erase(orders_.begin() + bi);
    }
}

void ReferenceMatcher::new_order(const InputEvent& in, std::vector<OutputEvent>& out) {
    auto reject = [&](RejectReason r) {
        OutputEvent e = ev(OutputKind::Rejected, seq_, in.id);
        e.reject = r;
        out.push_back(e);
    };
    const bool market = in.type == OrderType::Market;
    if (in.id == 0) return reject(RejectReason::InvalidId);
    if (find(in.id) >= 0) return reject(RejectReason::DuplicateId);
    if (in.qty <= 0 || in.qty > kMaxQty) return reject(RejectReason::InvalidQty);
    if (!market && (in.price <= 0 || in.price > kMaxPrice)) return reject(RejectReason::InvalidPrice);
    if (in.type == OrderType::PostOnly) {
        const int bi = best_opposite(in.side);
        if (bi >= 0 && can_trade(in.side, in.price, orders_[static_cast<std::size_t>(bi)].price))
            return reject(RejectReason::PostOnlyWouldCross);
    }
    const Price px = market ? 0 : in.price;
    OutputEvent a = ev(OutputKind::Accepted, seq_, in.id);
    a.side = in.side;
    a.price = px;
    a.qty = in.qty;
    out.push_back(a);

    auto kill = [&](Qty q, CancelReason why) {
        OutputEvent e = ev(OutputKind::Canceled, seq_, in.id);
        e.side = in.side;
        e.price = px;
        e.qty = q;
        e.cancel = why;
        out.push_back(e);
    };
    if (in.type == OrderType::FOK) {
        Qty avail = 0;
        for (const Resting& o : orders_)
            if (o.side != in.side && can_trade(in.side, px, o.price)) avail += o.qty;
        if (avail < in.qty) return kill(in.qty, CancelReason::FokUnfilled);
    }
    Qty q = in.qty;
    do_match(in.id, in.side, !market, px, q, out);
    if (q == 0) return;
    if (in.type == OrderType::Limit || in.type == OrderType::PostOnly)
        orders_.push_back(Resting{in.id, in.side, px, q, clock_++});
    else
        kill(q, CancelReason::Unfilled);
}

void ReferenceMatcher::cancel(const InputEvent& in, std::vector<OutputEvent>& out) {
    const int i = in.id == 0 ? -1 : find(in.id);
    if (i < 0) {
        OutputEvent e = ev(OutputKind::Rejected, seq_, in.id);
        e.reject = RejectReason::UnknownId;
        out.push_back(e);
        return;
    }
    const Resting o = orders_[static_cast<std::size_t>(i)];
    orders_.erase(orders_.begin() + i);
    OutputEvent e = ev(OutputKind::Canceled, seq_, o.id);
    e.side = o.side;
    e.price = o.price;
    e.qty = o.qty;
    e.cancel = CancelReason::User;
    out.push_back(e);
}

void ReferenceMatcher::modify(const InputEvent& in, std::vector<OutputEvent>& out) {
    auto reject = [&](RejectReason r) {
        OutputEvent e = ev(OutputKind::Rejected, seq_, in.id);
        e.reject = r;
        out.push_back(e);
    };
    const int i = in.id == 0 ? -1 : find(in.id);
    if (i < 0) return reject(RejectReason::UnknownId);
    if (in.qty <= 0 || in.qty > kMaxQty) return reject(RejectReason::InvalidQty);
    if (in.price <= 0 || in.price > kMaxPrice) return reject(RejectReason::InvalidPrice);
    Resting& o = orders_[static_cast<std::size_t>(i)];
    OutputEvent m = ev(OutputKind::Modified, seq_, o.id);
    m.side = o.side;
    m.price = in.price;
    m.qty = in.qty;
    out.push_back(m);
    if (in.price == o.price && in.qty <= o.qty) {
        o.qty = in.qty; // keeps its time stamp, so keeps its place
        return;
    }
    const Resting old = o;
    orders_.erase(orders_.begin() + i);
    Qty q = in.qty;
    do_match(old.id, old.side, true, in.price, q, out);
    if (q > 0) orders_.push_back(Resting{old.id, old.side, in.price, q, clock_++});
}

Depth ReferenceMatcher::depth(Side s) const {
    std::map<Price, Qty> m;
    for (const Resting& o : orders_)
        if (o.side == s) m[o.price] += o.qty;
    Depth d(m.begin(), m.end());
    if (s == Side::Buy) std::reverse(d.begin(), d.end());
    return d;
}

std::vector<OrderView> ReferenceMatcher::orders_in_priority(Side s) const {
    std::vector<Resting> v;
    for (const Resting& o : orders_)
        if (o.side == s) v.push_back(o);
    std::sort(v.begin(), v.end(), [s](const Resting& a, const Resting& b) {
        if (a.price != b.price) return s == Side::Buy ? a.price > b.price : a.price < b.price;
        return a.time < b.time;
    });
    std::vector<OrderView> r;
    for (const Resting& o : v) r.push_back(OrderView{o.id, o.side, o.price, o.qty});
    return r;
}

} // namespace exch
