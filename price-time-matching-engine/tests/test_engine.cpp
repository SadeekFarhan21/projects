// Hand-written scenarios for each rule. The differential test covers the
// combinations; these pin down the intended semantics one at a time.
#include "exchange/engine.hpp"
#include "exchange/market_data.hpp"

#include <gtest/gtest.h>

using namespace exch;

namespace {

struct Harness {
    MatchingEngine eng;
    std::vector<OutputEvent> last;

    const std::vector<OutputEvent>& send(const InputEvent& e) {
        last.clear();
        eng.process(e, last);
        eng.check_invariants();
        return last;
    }
    const std::vector<OutputEvent>& lim(OrderId id, Side s, Price p, Qty q) {
        return send(InputEvent::new_order(id, s, OrderType::Limit, p, q));
    }
    std::vector<OutputEvent> of(OutputKind k) const {
        std::vector<OutputEvent> v;
        for (auto& e : last)
            if (e.kind == k) v.push_back(e);
        return v;
    }
};

constexpr Side B = Side::Buy;
constexpr Side S = Side::Sell;

} // namespace

TEST(Engine, RestingLimitIsAcceptedAndPublished) {
    Harness h;
    auto& out = h.lim(1, B, 100, 10);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].kind, OutputKind::Accepted);
    EXPECT_EQ(out[1].kind, OutputKind::BookUpdate);
    EXPECT_EQ(out[1].side, B);
    EXPECT_EQ(out[1].price, 100);
    EXPECT_EQ(out[1].qty, 10);
    EXPECT_EQ(h.eng.best_price(B), 100);
    EXPECT_FALSE(h.eng.best_price(S).has_value());
}

TEST(Engine, PricePriorityBeatsTime) {
    Harness h;
    h.lim(1, S, 102, 5);
    h.lim(2, S, 101, 5); // better price, later
    h.lim(3, B, 102, 5);
    auto t = h.of(OutputKind::Trade);
    ASSERT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0].maker_id, 2u);
    EXPECT_EQ(t[0].price, 101); // trades at the maker's price
}

TEST(Engine, TimePriorityWithinLevel) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.lim(2, S, 100, 5);
    h.lim(3, S, 100, 5);
    h.lim(4, B, 100, 12);
    auto t = h.of(OutputKind::Trade);
    ASSERT_EQ(t.size(), 3u);
    EXPECT_EQ(t[0].maker_id, 1u);
    EXPECT_EQ(t[1].maker_id, 2u);
    EXPECT_EQ(t[2].maker_id, 3u);
    EXPECT_EQ(t[2].qty, 2);
    EXPECT_EQ(h.eng.find_order(3)->qty, 3);
    EXPECT_FALSE(h.eng.find_order(4).has_value()); // taker fully filled
}

TEST(Engine, SweepsLevelsAndRestsRemainder) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.lim(2, S, 101, 5);
    h.lim(3, S, 105, 5);
    h.lim(4, B, 102, 15);
    auto t = h.of(OutputKind::Trade);
    ASSERT_EQ(t.size(), 2u);
    EXPECT_EQ(h.eng.best_price(B), 102);
    EXPECT_EQ(h.eng.find_order(4)->qty, 5);
    EXPECT_EQ(h.eng.best_price(S), 105);
    // L2: both ask levels deleted, bid level created. Sorted bids then asks.
    auto l2 = h.of(OutputKind::BookUpdate);
    ASSERT_EQ(l2.size(), 3u);
    EXPECT_EQ(l2[0].side, B);
    EXPECT_EQ(l2[0].qty, 5);
    EXPECT_EQ(l2[1].price, 100);
    EXPECT_EQ(l2[1].qty, 0);
    EXPECT_EQ(l2[2].price, 101);
    EXPECT_EQ(l2[2].qty, 0);
}

TEST(Engine, TradeIdsAreSequential) {
    Harness h;
    h.lim(1, S, 100, 1);
    h.lim(2, S, 100, 1);
    h.lim(3, B, 100, 2);
    auto t = h.of(OutputKind::Trade);
    ASSERT_EQ(t.size(), 2u);
    EXPECT_EQ(t[0].trade_id, 1u);
    EXPECT_EQ(t[1].trade_id, 2u);
}

TEST(Engine, MarketOrderTakesAnyPriceAndCancelsRemainder) {
    Harness h;
    h.lim(1, B, 90, 5);
    h.lim(2, B, 80, 5);
    h.send(InputEvent::new_order(3, S, OrderType::Market, 0, 20));
    EXPECT_EQ(h.of(OutputKind::Trade).size(), 2u);
    auto c = h.of(OutputKind::Canceled);
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0].qty, 10);
    EXPECT_EQ(c[0].cancel, CancelReason::Unfilled);
    EXPECT_EQ(h.eng.order_count(), 0u);
}

TEST(Engine, IocRespectsLimitAndNeverRests) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.lim(2, S, 103, 5);
    h.send(InputEvent::new_order(3, B, OrderType::IOC, 101, 8));
    auto t = h.of(OutputKind::Trade);
    ASSERT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0].qty, 5);
    EXPECT_EQ(h.of(OutputKind::Canceled).at(0).qty, 3);
    EXPECT_FALSE(h.eng.find_order(3).has_value());
    EXPECT_EQ(h.eng.best_price(S), 103);
}

TEST(Engine, FokAllOrNothing) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.lim(2, S, 101, 5);
    // Needs 11 at <= 101, only 10 available: nothing trades, book untouched.
    auto& out = h.send(InputEvent::new_order(3, B, OrderType::FOK, 101, 11));
    EXPECT_EQ(h.of(OutputKind::Trade).size(), 0u);
    EXPECT_EQ(h.of(OutputKind::Canceled).at(0).cancel, CancelReason::FokUnfilled);
    EXPECT_EQ(h.of(OutputKind::BookUpdate).size(), 0u);
    EXPECT_EQ(out.size(), 2u); // Accepted, Canceled
    EXPECT_EQ(h.eng.order_count(), 2u);
    // Exactly 10 fills completely.
    h.send(InputEvent::new_order(4, B, OrderType::FOK, 101, 10));
    EXPECT_EQ(h.of(OutputKind::Trade).size(), 2u);
    EXPECT_EQ(h.of(OutputKind::Canceled).size(), 0u);
    EXPECT_EQ(h.eng.order_count(), 0u);
}

TEST(Engine, FokIgnoresLiquidityBeyondLimit) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.lim(2, S, 110, 50);
    h.send(InputEvent::new_order(3, B, OrderType::FOK, 105, 6));
    EXPECT_EQ(h.of(OutputKind::Trade).size(), 0u);
}

TEST(Engine, PostOnlyRejectsWhenCrossingAndRestsOtherwise) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.send(InputEvent::new_order(2, B, OrderType::PostOnly, 100, 5));
    ASSERT_EQ(h.last.size(), 1u);
    EXPECT_EQ(h.last[0].kind, OutputKind::Rejected);
    EXPECT_EQ(h.last[0].reject, RejectReason::PostOnlyWouldCross);
    h.send(InputEvent::new_order(3, B, OrderType::PostOnly, 99, 5));
    EXPECT_EQ(h.last[0].kind, OutputKind::Accepted);
    EXPECT_EQ(h.eng.best_price(B), 99);
}

TEST(Engine, CancelRemovesAndReportsOpenQty) {
    Harness h;
    h.lim(1, B, 100, 10);
    h.lim(2, S, 100, 4);
    h.send(InputEvent::cancel(1));
    auto c = h.of(OutputKind::Canceled);
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0].qty, 6);
    EXPECT_EQ(c[0].cancel, CancelReason::User);
    EXPECT_EQ(h.of(OutputKind::BookUpdate).at(0).qty, 0);
    EXPECT_EQ(h.eng.order_count(), 0u);
}

TEST(Engine, CancelUnknownOrFilledIsRejected) {
    Harness h;
    h.send(InputEvent::cancel(42));
    EXPECT_EQ(h.last.at(0).reject, RejectReason::UnknownId);
    h.lim(1, B, 100, 1);
    h.lim(2, S, 100, 1);
    h.send(InputEvent::cancel(1));
    EXPECT_EQ(h.last.at(0).reject, RejectReason::UnknownId);
}

TEST(Engine, ModifyDownKeepsPriority) {
    Harness h;
    h.lim(1, S, 100, 10);
    h.lim(2, S, 100, 10);
    h.send(InputEvent::modify(1, 100, 4));
    EXPECT_EQ(h.last.at(0).kind, OutputKind::Modified);
    EXPECT_EQ(h.of(OutputKind::BookUpdate).at(0).qty, 14);
    h.lim(3, B, 100, 4);
    EXPECT_EQ(h.of(OutputKind::Trade).at(0).maker_id, 1u);
}

TEST(Engine, ModifyUpLosesPriority) {
    Harness h;
    h.lim(1, S, 100, 10);
    h.lim(2, S, 100, 10);
    h.send(InputEvent::modify(1, 100, 11));
    h.lim(3, B, 100, 4);
    EXPECT_EQ(h.of(OutputKind::Trade).at(0).maker_id, 2u);
    auto q = h.eng.orders_in_priority(S);
    ASSERT_EQ(q.size(), 2u);
    EXPECT_EQ(q[0].id, 2u);
    EXPECT_EQ(q[1].id, 1u);
}

TEST(Engine, ModifyPriceLosesPriorityAndCanCross) {
    Harness h;
    h.lim(1, B, 99, 10);
    h.lim(2, S, 101, 3);
    h.send(InputEvent::modify(1, 101, 10));
    auto t = h.of(OutputKind::Trade);
    ASSERT_EQ(t.size(), 1u);
    EXPECT_EQ(t[0].id, 1u); // the modified order is the taker
    EXPECT_EQ(t[0].qty, 3);
    EXPECT_EQ(h.eng.find_order(1)->qty, 7);
    EXPECT_EQ(h.eng.find_order(1)->price, 101);
}

TEST(Engine, ModifyValidation) {
    Harness h;
    h.lim(1, B, 99, 10);
    h.send(InputEvent::modify(1, 99, 0));
    EXPECT_EQ(h.last.at(0).reject, RejectReason::InvalidQty);
    h.send(InputEvent::modify(1, 0, 5));
    EXPECT_EQ(h.last.at(0).reject, RejectReason::InvalidPrice);
    h.send(InputEvent::modify(7, 99, 5));
    EXPECT_EQ(h.last.at(0).reject, RejectReason::UnknownId);
    EXPECT_EQ(h.eng.find_order(1)->qty, 10);
}

TEST(Engine, NewOrderValidation) {
    Harness h;
    h.lim(0, B, 100, 1);
    EXPECT_EQ(h.last.at(0).reject, RejectReason::InvalidId);
    h.lim(1, B, 100, 0);
    EXPECT_EQ(h.last.at(0).reject, RejectReason::InvalidQty);
    h.lim(2, B, -1, 1);
    EXPECT_EQ(h.last.at(0).reject, RejectReason::InvalidPrice);
    h.lim(3, B, kMaxPrice + 1, 1);
    EXPECT_EQ(h.last.at(0).reject, RejectReason::InvalidPrice);
    h.lim(4, B, 100, kMaxQty + 1);
    EXPECT_EQ(h.last.at(0).reject, RejectReason::InvalidQty);
    h.lim(5, B, 100, 1);
    h.lim(5, S, 200, 1);
    EXPECT_EQ(h.last.at(0).reject, RejectReason::DuplicateId);
    EXPECT_EQ(h.eng.order_count(), 1u);
}

TEST(Engine, IdCanBeReusedAfterOrderIsGone) {
    // v0 scopes id uniqueness to live orders only.
    Harness h;
    h.lim(1, B, 100, 1);
    h.send(InputEvent::cancel(1));
    h.lim(1, B, 101, 1);
    EXPECT_EQ(h.last.at(0).kind, OutputKind::Accepted);
}

TEST(Engine, BookUpdatesAreCoalescedPerInput) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.lim(2, S, 100, 5);
    h.lim(3, S, 100, 5);
    h.lim(4, B, 100, 11); // three fills at one level, one L2 update
    auto l2 = h.of(OutputKind::BookUpdate);
    ASSERT_EQ(l2.size(), 1u);
    EXPECT_EQ(l2[0].qty, 4);
}

TEST(Engine, NoOpModifyPublishesNoL2) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.send(InputEvent::modify(1, 100, 5));
    EXPECT_EQ(h.of(OutputKind::Modified).size(), 1u);
    EXPECT_EQ(h.of(OutputKind::BookUpdate).size(), 0u);
}

TEST(Engine, SeqNumbersTagEveryOutput) {
    Harness h;
    h.lim(1, S, 100, 5);
    h.lim(2, B, 100, 5);
    for (auto& e : h.last) EXPECT_EQ(e.seq, 1u);
    EXPECT_EQ(h.eng.next_seq(), 2u);
}

TEST(Engine, ManyLevelsInsertAndDrainInOrder) {
    Harness h;
    // Insert out of order, including far from the touch, to exercise both the
    // short linear scan and the binary search in find_level.
    const Price prices[] = {150, 120, 180, 101, 199, 110, 160, 130, 170, 140, 190, 105, 125, 175, 115};
    OrderId id = 1;
    for (Price p : prices) h.lim(id++, S, p, 1);
    EXPECT_EQ(h.eng.level_count(S), 15u);
    Price last = 0;
    for (int i = 0; i < 15; ++i) {
        h.send(InputEvent::new_order(id++, B, OrderType::Market, 0, 1));
        auto t = h.of(OutputKind::Trade);
        ASSERT_EQ(t.size(), 1u);
        EXPECT_GT(t[0].price, last);
        last = t[0].price;
    }
    EXPECT_EQ(h.eng.order_count(), 0u);
}

TEST(MarketData, MirrorTracksBookAndTape) {
    Harness h;
    MarketDataFeed md;
    auto feed = [&] {
        for (auto& e : h.last) md.on_event(e);
    };
    h.lim(1, S, 101, 5); feed();
    h.lim(2, S, 102, 5); feed();
    h.lim(3, B, 99, 7); feed();
    h.lim(4, B, 101, 2); feed();
    EXPECT_EQ(md.depth(S), h.eng.depth(S));
    EXPECT_EQ(md.depth(B), h.eng.depth(B));
    EXPECT_EQ(md.best(S), 101);
    ASSERT_EQ(md.tape().size(), 1u);
    EXPECT_EQ(md.tape()[0].qty, 2);
    EXPECT_EQ(md.traded_volume(), 2);
}
