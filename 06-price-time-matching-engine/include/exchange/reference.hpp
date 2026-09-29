// Reference matcher: the slowest obviously-correct implementation of the
// same rules as MatchingEngine. Resting orders live in one flat vector and
// every decision is a linear scan. L2 updates are produced by diffing a full
// book snapshot before and after each input, so it shares no incremental
// bookkeeping with the real engine. Used only as a test oracle.
#pragma once

#include "exchange/engine.hpp"
#include "exchange/events.hpp"

#include <map>
#include <vector>

namespace exch {

class ReferenceMatcher {
public:
    void process(const InputEvent& in, std::vector<OutputEvent>& out);

    Depth depth(Side s) const;
    std::vector<OrderView> orders_in_priority(Side s) const;
    std::size_t order_count() const { return orders_.size(); }

private:
    struct Resting {
        OrderId id;
        Side side;
        Price price;
        Qty qty;
        std::uint64_t time; // arrival stamp; lower is earlier
    };

    using Snapshot = std::map<std::pair<int, Price>, Qty>;
    Snapshot snapshot() const;
    int find(OrderId id) const;
    int best_opposite(Side taker_side) const; // index of next maker, or -1
    void do_match(OrderId taker, Side side, bool has_limit, Price limit, Qty& qty, std::vector<OutputEvent>& out);
    void new_order(const InputEvent& in, std::vector<OutputEvent>& out);
    void cancel(const InputEvent& in, std::vector<OutputEvent>& out);
    void modify(const InputEvent& in, std::vector<OutputEvent>& out);

    std::vector<Resting> orders_;
    std::uint64_t clock_{0};
    SeqNo seq_{0};
    TradeId next_trade_{1};
};

} // namespace exch
