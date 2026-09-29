// Market-data side of the event stream. A MarketDataFeed is a downstream
// consumer: it sees only output events and rebuilds the public L2 view and
// a trade tape from them. If the engine forgot to publish a level change,
// this mirror drifts from the real book, which the tests check for.
#pragma once

#include "exchange/engine.hpp"
#include "exchange/events.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <vector>

namespace exch {

struct TradePrint {
    SeqNo seq;
    TradeId id;
    Side aggressor;
    Price price;
    Qty qty;
};

class MarketDataFeed {
public:
    // Consume one output event. Non market-data events are ignored.
    void on_event(const OutputEvent& e);

    Depth depth(Side s, std::size_t max_levels = SIZE_MAX) const;
    std::optional<Price> best(Side s) const;
    const std::vector<TradePrint>& tape() const { return tape_; }
    Qty traded_volume() const { return volume_; }
    std::uint64_t book_updates() const { return updates_; }

    // Optional subscriber callbacks, called after the mirror is updated.
    std::function<void(const OutputEvent&)> on_l2;
    std::function<void(const TradePrint&)> on_trade;

    bool keep_tape = true;

private:
    std::map<Price, Qty, std::greater<Price>> bids_; // best first
    std::map<Price, Qty> asks_;                      // best first
    std::vector<TradePrint> tape_;
    Qty volume_{0};
    std::uint64_t updates_{0};
};

// FNV-1a over the fields of every output event. Two runs agree on the
// digest iff (with overwhelming probability) they produced the same stream.
class StreamDigest {
public:
    void add(const OutputEvent& e);
    std::uint64_t value() const { return h_; }
    std::uint64_t count() const { return n_; }

private:
    void mix(std::uint64_t v);
    std::uint64_t h_{1469598103934665603ULL};
    std::uint64_t n_{0};
};

} // namespace exch
