// The tick-to-trade pipeline.
//
// Threaded layout (see DESIGN.md for the diagram):
//
//   feed ──md_q──> trader (book + strategy + risk) ──ord_q──> gateway ──wire_q──> exchange sim
//
// run_threaded<Book, Queue, Heap> is templated on every knob the ablation
// flips: book implementation, queue implementation, and whether messages are
// passed by value in ring slots (Heap = false) or as heap pointers allocated
// by the producer and freed by the consumer (Heap = true).
//
// run_inline<Book> does feed, book, strategy, risk and gateway on one thread
// and only hands the encoded frame to the exchange thread.
//
// Timestamps: t_in is taken when the feed handler picks a message up (after
// pacing, before decode). t_out is taken by the gateway once the order frame
// is encoded, immediately before it is published on the exchange connection.
// Tick-to-trade = t_out - t_in.
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "llt/capture.hpp"
#include "llt/clock.hpp"
#include "llt/itch.hpp"
#include "llt/messages.hpp"
#include "llt/order_book.hpp"
#include "llt/ouch.hpp"
#include "llt/queues.hpp"
#include "llt/risk.hpp"
#include "llt/strategy.hpp"
#include "llt/thread_util.hpp"

namespace llt {

inline constexpr std::size_t kMdQueueCap = 1u << 14;
inline constexpr std::size_t kOrderQueueCap = 1u << 12;
inline constexpr std::size_t kWireQueueCap = 1u << 12;

struct PipelineConfig {
    bool pace{true};    // replay at capture timestamps (else as fast as possible)
    double speed{1.0};  // pacing multiplier: 2.0 replays twice as fast
    Qos qos{Qos::Interactive};
    bool try_affinity{true};
    StrategyParams strategy{};
    RiskLimits risk{};
    uint32_t book_tick{100};
    uint32_t book_levels{4096};
    std::size_t order_capacity{1u << 21};
    // Test hooks, called on the launching thread right before the worker
    // threads are released and right after they are joined.
    std::function<void()> on_hot_start;
    std::function<void()> on_hot_end;
};

struct RunResult {
    std::size_t md_messages{0};  // decoded messages the feed handler forwarded
    std::size_t orders{0};       // orders the exchange received
    std::array<uint64_t, static_cast<std::size_t>(RiskVerdict::Count)> verdicts{};
    std::vector<OrderTiming> timings;          // [orders], indexed by client_id
    std::vector<ouch::EnterOrder> exch_orders; // [orders], as decoded by the exchange
    uint64_t seq_errors{0};
    uint64_t decode_errors{0};
    uint64_t backpressure_spins{0};  // failed pushes into md_q (feed blocked)
    double wall_ns{0};
    BookStats book{};
    std::array<PlacementResult, 4> placement{};
};

// Book + strategy + risk. Shared by the threaded and inline pipelines so both
// make identical decisions for identical input.
template <class Book>
class TradingCore {
public:
    explicit TradingCore(const PipelineConfig& c)
        : book_(std::make_unique<Book>(c.book_tick, c.book_levels, c.order_capacity)),
          strat_(c.strategy),
          risk_(c.risk) {}

    bool on_event(const MdEvent& e, OrderRequest& out) noexcept {
        if (e.type == 'R') {
            strat_.on_directory(e);
            return false;
        }
        book_->apply(e);
        const Top t = book_->top(e.locate);
        if (!strat_.on_book(e, t, out)) return false;
        const RiskVerdict v = risk_.check(out, t, e.exch_ts);
        ++verdicts_[static_cast<std::size_t>(v)];
        if (v != RiskVerdict::Accept) return false;
        out.client_id = next_id_++;
        out.t_decide = clk::now();
        return true;
    }

    const Book& book() const { return *book_; }
    const auto& verdicts() const { return verdicts_; }

private:
    std::unique_ptr<Book> book_;
    ImbalanceStrategy strat_;
    RiskEngine risk_;
    uint32_t next_id_{0};
    std::array<uint64_t, static_cast<std::size_t>(RiskVerdict::Count)> verdicts_{};
};

namespace detail {

// Replays the capture, calling sink(MdEvent) for each decodable message.
template <class Sink>
std::size_t feed_loop(const Capture& cap, const PipelineConfig& cfg, Sink&& sink) {
    itch::FrameCursor cur{cap.bytes.data(), cap.bytes.data() + cap.bytes.size()};
    const double ticks_per_ns = static_cast<double>(clk::freq_hz()) / 1e9 / cfg.speed;
    bool first = true;
    uint64_t ts0 = 0, start = 0;
    std::size_t n = 0;
    const uint8_t* msg;
    uint16_t len;
    while (cur.next(msg, len)) {
        if (cfg.pace && len >= itch::kHeaderLen) {
            const uint64_t ts = itch::be48(msg + 5);
            if (first) {
                ts0 = ts;
                start = clk::now();
                first = false;
            }
            const uint64_t target =
                start + static_cast<uint64_t>(static_cast<double>(ts - ts0) * ticks_per_ns);
            while (clk::now_unordered() < target) cpu_relax();
        }
        MdEvent e;
        e.t_in = clk::now();
        if (!itch::decode(msg, len, e)) continue;
        sink(e);
        ++n;
    }
    return n;
}

// The simulated exchange: reads frames off the connection, decodes, checks the
// token sequence, and records receive time. Stops on a zero-length frame.
template <class WireQ>
void exchange_loop(WireQ& wire_q, RunResult& res, std::vector<uint64_t>& t_rx) {
    WireFrame f;
    std::size_t n = 0;
    for (;;) {
        pop_spin(wire_q, f);
        if (f.len == 0) break;
        const uint64_t t = clk::now();
        ouch::EnterOrder eo;
        if (!ouch::decode_enter(f.data, f.len, eo)) {
            ++res.decode_errors;
            continue;
        }
        if (eo.token != n) ++res.seq_errors;
        if (n < res.exch_orders.size()) {
            res.exch_orders[n] = eo;
            t_rx[n] = t;
        }
        ++n;
    }
    res.orders = n;
}

inline void finish(RunResult& res, const std::vector<uint64_t>& t_rx) {
    const std::size_t n = std::min(res.orders, res.timings.size());
    for (std::size_t i = 0; i < n; ++i) res.timings[i].t_rx = t_rx[i];
    res.timings.resize(n);
    res.exch_orders.resize(n);
}

inline void prepare(RunResult& res, std::vector<uint64_t>& t_rx, std::size_t max_orders) {
    // assign() writes every element, which also pre-faults the pages so the
    // hot path never takes a first-touch page fault.
    res.timings.assign(max_orders, OrderTiming{});
    res.exch_orders.assign(max_orders, ouch::EnterOrder{});
    t_rx.assign(max_orders, 0);
}

}  // namespace detail

template <class Book, template <class, std::size_t> class Q, bool Heap>
RunResult run_threaded(const Capture& cap, const PipelineConfig& cfg) {
    using MdMsg = std::conditional_t<Heap, MdEvent*, MdEvent>;
    using OrdMsg = std::conditional_t<Heap, OrderRequest*, OrderRequest>;
    auto md_q = std::make_unique<Q<MdMsg, kMdQueueCap>>();
    auto ord_q = std::make_unique<Q<OrdMsg, kOrderQueueCap>>();
    auto wire_q = std::make_unique<Q<WireFrame, kWireQueueCap>>();
    TradingCore<Book> core(cfg);

    RunResult res;
    std::vector<uint64_t> t_rx;
    detail::prepare(res, t_rx, cap.message_count + 1);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    auto enter = [&](int idx, const char* name) {
        set_thread_name(name);
        res.placement[idx] = place_current_thread(cfg.qos, cfg.try_affinity ? idx : -1);
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!go.load(std::memory_order_acquire)) cpu_relax();
    };

    std::thread feed([&] {
        enter(0, "llt-feed");
        uint64_t bp = 0;
        auto push = [&](const MdEvent& e) {
            if constexpr (Heap) bp += push_spin(*md_q, new MdEvent(e));
            else bp += push_spin(*md_q, e);
        };
        res.md_messages = detail::feed_loop(cap, cfg, push);
        MdEvent z;
        z.type = 'Z';
        push(z);
        res.backpressure_spins = bp;
    });

    std::thread trader([&] {
        enter(1, "llt-trader");
        MdMsg m;
        OrderRequest o;
        auto push = [&](const OrderRequest& r) {
            if constexpr (Heap) push_spin(*ord_q, new OrderRequest(r));
            else push_spin(*ord_q, r);
        };
        for (;;) {
            pop_spin(*md_q, m);
            MdEvent e;
            if constexpr (Heap) {
                e = *m;
                delete m;
            } else {
                e = m;
            }
            if (e.type == 'Z') break;
            if (core.on_event(e, o)) push(o);
        }
        OrderRequest z;
        z.side = 'Z';
        push(z);
    });

    std::thread gateway([&] {
        enter(2, "llt-gateway");
        OrdMsg m;
        WireFrame f;
        for (;;) {
            pop_spin(*ord_q, m);
            OrderRequest o;
            if constexpr (Heap) {
                o = *m;
                delete m;
            } else {
                o = m;
            }
            if (o.side == 'Z') break;
            f.len = static_cast<uint16_t>(ouch::encode_enter(f.data, o));
            // Stamp before publishing: once the frame is visible the exchange thread
            // may stamp t_rx before this thread gets to read the clock.
            const uint64_t t_out = clk::now();
            push_spin(*wire_q, f);
            if (o.client_id < res.timings.size())
                res.timings[o.client_id] = OrderTiming{o.t_in, o.t_decide, t_out, 0};
        }
        f.len = 0;
        push_spin(*wire_q, f);
    });

    std::thread exchange([&] {
        enter(3, "llt-exchange");
        detail::exchange_loop(*wire_q, res, t_rx);
    });

    while (ready.load(std::memory_order_acquire) < 4) std::this_thread::yield();
    if (cfg.on_hot_start) cfg.on_hot_start();
    const uint64_t t0 = clk::now();
    go.store(true, std::memory_order_release);
    feed.join();
    trader.join();
    gateway.join();
    exchange.join();
    res.wall_ns = clk::to_ns(clk::now() - t0);
    if (cfg.on_hot_end) cfg.on_hot_end();

    res.verdicts = core.verdicts();
    res.book = core.book().stats();
    detail::finish(res, t_rx);
    return res;
}

template <class Book>
RunResult run_inline(const Capture& cap, const PipelineConfig& cfg) {
    auto wire_q = std::make_unique<SpscRing<WireFrame, kWireQueueCap>>();
    TradingCore<Book> core(cfg);
    RunResult res;
    std::vector<uint64_t> t_rx;
    detail::prepare(res, t_rx, cap.message_count + 1);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    auto enter = [&](int idx, const char* name) {
        set_thread_name(name);
        res.placement[idx] = place_current_thread(cfg.qos, cfg.try_affinity ? idx : -1);
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!go.load(std::memory_order_acquire)) cpu_relax();
    };

    std::thread hot([&] {
        enter(0, "llt-inline");
        OrderRequest o;
        WireFrame f;
        res.md_messages = detail::feed_loop(cap, cfg, [&](const MdEvent& e) {
            if (!core.on_event(e, o)) return;
            f.len = static_cast<uint16_t>(ouch::encode_enter(f.data, o));
            // Stamp before publishing: once the frame is visible the exchange thread
            // may stamp t_rx before this thread gets to read the clock.
            const uint64_t t_out = clk::now();
            push_spin(*wire_q, f);
            if (o.client_id < res.timings.size())
                res.timings[o.client_id] = OrderTiming{o.t_in, o.t_decide, t_out, 0};
        });
        f.len = 0;
        push_spin(*wire_q, f);
    });
    std::thread exchange([&] {
        enter(3, "llt-exchange");
        detail::exchange_loop(*wire_q, res, t_rx);
    });

    while (ready.load(std::memory_order_acquire) < 2) std::this_thread::yield();
    if (cfg.on_hot_start) cfg.on_hot_start();
    const uint64_t t0 = clk::now();
    go.store(true, std::memory_order_release);
    hot.join();
    exchange.join();
    res.wall_ns = clk::to_ns(clk::now() - t0);
    if (cfg.on_hot_end) cfg.on_hot_end();

    res.verdicts = core.verdicts();
    res.book = core.book().stats();
    detail::finish(res, t_rx);
    return res;
}

// Tick-to-trade samples in nanoseconds.
inline std::vector<double> tick_to_trade_ns(const RunResult& r) {
    std::vector<double> v;
    v.reserve(r.timings.size());
    for (const auto& t : r.timings) v.push_back(clk::to_ns(t.t_out - t.t_in));
    return v;
}

}  // namespace llt
