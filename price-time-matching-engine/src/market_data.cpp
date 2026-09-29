#include "exchange/market_data.hpp"

namespace exch {

void MarketDataFeed::on_event(const OutputEvent& e) {
    if (e.kind == OutputKind::BookUpdate) {
        ++updates_;
        if (e.side == Side::Buy) {
            if (e.qty == 0) bids_.erase(e.price);
            else bids_[e.price] = e.qty;
        } else {
            if (e.qty == 0) asks_.erase(e.price);
            else asks_[e.price] = e.qty;
        }
        if (on_l2) on_l2(e);
    } else if (e.kind == OutputKind::Trade) {
        const TradePrint t{e.seq, e.trade_id, e.side, e.price, e.qty};
        volume_ += e.qty;
        if (keep_tape) tape_.push_back(t);
        if (on_trade) on_trade(t);
    }
}

Depth MarketDataFeed::depth(Side s, std::size_t max_levels) const {
    Depth d;
    auto fill = [&](const auto& m) {
        for (const auto& kv : m) {
            if (d.size() >= max_levels) break;
            d.emplace_back(kv.first, kv.second);
        }
    };
    if (s == Side::Buy) fill(bids_);
    else fill(asks_);
    return d;
}

std::optional<Price> MarketDataFeed::best(Side s) const {
    if (s == Side::Buy) return bids_.empty() ? std::nullopt : std::optional<Price>(bids_.begin()->first);
    return asks_.empty() ? std::nullopt : std::optional<Price>(asks_.begin()->first);
}

void StreamDigest::mix(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h_ ^= (v >> (8 * i)) & 0xFF;
        h_ *= 1099511628211ULL;
    }
}

void StreamDigest::add(const OutputEvent& e) {
    ++n_;
    mix(static_cast<std::uint64_t>(e.kind));
    mix(e.seq);
    mix(e.id);
    mix(e.maker_id);
    mix(e.trade_id);
    mix(static_cast<std::uint64_t>(e.side));
    mix(static_cast<std::uint64_t>(e.price));
    mix(static_cast<std::uint64_t>(e.qty));
    mix(static_cast<std::uint64_t>(e.reject));
    mix(static_cast<std::uint64_t>(e.cancel));
}

} // namespace exch
