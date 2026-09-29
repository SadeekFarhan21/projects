// Latency and throughput benchmark for the matching engine.
//
// Each path (add, cancel, match) is measured in a steady state: the book
// starts with `book_orders` resting orders and every timed operation is
// followed by an untimed operation that restores the book size, so the
// state being measured does not drift over the run.
//
//   bench_engine [--variant NAME] [--n OPS] [--sizes 1000,10000,...] [--flow N] [--out DIR]
//
// Outputs (appended, one row per scenario):
//   DIR/latency_<variant>.csv   percentiles per (scenario, book size)
//   DIR/cdf_<variant>.csv       1000 quantiles per scenario at the reference size
//   DIR/throughput_<variant>.csv  un-instrumented replay of the random flow
#include "exchange/engine.hpp"
#include "exchange/order_flow.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

using namespace exch;
using Clock = std::chrono::steady_clock;

namespace {

// CPU time consumed by this thread. On a shared machine wall time includes
// time spent descheduled; CPU time does not, so throughput is reported both ways.
inline std::int64_t thread_cpu_ns() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// Keep the optimizer from discarding work whose result we never read.
template <class T>
inline void do_not_optimize(T const& v) {
    asm volatile("" : : "r,m"(v) : "memory");
}

struct Pct {
    double mean, p50, p90, p99, p999, max;
};

Pct percentiles(std::vector<std::int64_t>& v) {
    std::sort(v.begin(), v.end());
    auto at = [&](double q) {
        const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())));
        return static_cast<double>(v[i]);
    };
    const double mean = static_cast<double>(std::accumulate(v.begin(), v.end(), std::int64_t{0})) /
                        static_cast<double>(v.size());
    return {mean, at(0.50), at(0.90), at(0.99), at(0.999), static_cast<double>(v.back())};
}

// Tracks which ids are resting so the bench can cancel a random live order
// in O(1). Pure bookkeeping, always outside the timed region.
struct LiveSet {
    std::vector<OrderId> ids;
    std::unordered_map<OrderId, std::size_t> pos;
    void add(OrderId id) {
        pos[id] = ids.size();
        ids.push_back(id);
    }
    void remove(OrderId id) {
        auto it = pos.find(id);
        if (it == pos.end()) return;
        const std::size_t i = it->second;
        pos[ids.back()] = i;
        ids[i] = ids.back();
        ids.pop_back();
        pos.erase(it);
    }
};

constexpr Price kMid = 100'000;
constexpr Price kLevels = 100; // resting prices occupy this many levels per side
constexpr Qty kLot = 10;

struct Bench {
    MatchingEngine eng;
    LiveSet live;
    std::vector<OutputEvent> out;
    Rng rng;
    OrderId next_id = 1;

    explicit Bench(std::size_t book_orders, std::uint64_t seed) : eng(book_orders * 2 + 1024), rng(seed) {
        out.reserve(256);
        for (std::size_t i = 0; i < book_orders; ++i) add_resting(i % 2 == 0 ? Side::Buy : Side::Sell);
    }

    // Bids occupy [kMid - kLevels, kMid - 1], asks [kMid + 1, kMid + kLevels].
    Price resting_price(Side s) {
        const Price off = rng.range(1, kLevels);
        return s == Side::Buy ? kMid - off : kMid + off;
    }

    void apply(const InputEvent& e) {
        out.clear();
        eng.process(e, out);
        for (const OutputEvent& o : out) {
            if (o.kind == OutputKind::Trade) {
                // Maker fully filled when its trade qty equals a full lot; bench makers are all kLot.
                if (!eng.find_order(o.maker_id)) live.remove(o.maker_id);
            }
        }
    }

    InputEvent make_resting(Side s) {
        return InputEvent::new_order(next_id++, s, OrderType::Limit, resting_price(s), kLot);
    }

    void add_resting(Side s) {
        const InputEvent e = make_resting(s);
        apply(e);
        live.add(e.id);
    }

    OrderId random_live() {
        return live.ids[static_cast<std::size_t>(rng.range(0, static_cast<std::int64_t>(live.ids.size()) - 1))];
    }
};

void write_cdf(std::ofstream& f, const std::string& variant, const std::string& scen, std::size_t size,
               const std::vector<std::int64_t>& sorted) {
    for (int q = 0; q < 1000; ++q) {
        const std::size_t i = std::min(sorted.size() - 1, sorted.size() * static_cast<std::size_t>(q) / 1000);
        f << variant << ',' << scen << ',' << size << ',' << (q / 1000.0) << ',' << sorted[i] << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string variant = "idmap";
    std::string out_dir = "results";
    std::size_t n_ops = 1'000'000;
    std::size_t flow_events = 10'000'000;
    std::vector<std::size_t> sizes = {1'000, 10'000, 100'000, 1'000'000};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--variant") variant = v;
        else if (k == "--out") out_dir = v;
        else if (k == "--n") n_ops = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "--flow") flow_events = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "--sizes") {
            sizes.clear();
            std::stringstream ss(v);
            std::string tok;
            while (std::getline(ss, tok, ',')) sizes.push_back(std::strtoull(tok.c_str(), nullptr, 10));
        }
    }
#if defined(__APPLE__)
    // No thread pinning on macOS; the highest QoS class keeps us on P-cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    // Timer characterization: resolution and cost of one now() pair.
    {
        std::vector<std::int64_t> d;
        d.reserve(1'000'000);
        for (int i = 0; i < 1'000'000; ++i) {
            const auto a = now_ns();
            const auto b = now_ns();
            d.push_back(b - a);
        }
        std::int64_t min_step = INT64_MAX;
        for (int i = 0; i < 100000; ++i) {
            const auto a = now_ns();
            std::int64_t b;
            while ((b = now_ns()) == a) {}
            min_step = std::min(min_step, b - a);
        }
        Pct p = percentiles(d);
        std::printf("timer: back-to-back now() mean %.1f ns, p50 %.0f, p99 %.0f; smallest nonzero step %" PRId64
                    " ns\n",
                    p.mean, p.p50, p.p99, min_step);
        std::ofstream tf(out_dir + "/timer_" + variant + ".txt");
        tf << "back_to_back_mean_ns " << p.mean << "\nback_to_back_p50_ns " << p.p50 << "\nback_to_back_p99_ns "
           << p.p99 << "\nsmallest_nonzero_step_ns " << min_step << "\n";
    }

    std::ofstream lat(out_dir + "/latency_" + variant + ".csv");
    lat << "variant,scenario,book_orders,ops,mean_ns,p50_ns,p90_ns,p99_ns,p999_ns,max_ns\n";
    std::ofstream cdf(out_dir + "/cdf_" + variant + ".csv");
    cdf << "variant,scenario,book_orders,quantile,latency_ns\n";

    std::printf("%-8s %10s %9s %8s %8s %8s %8s %10s\n", "path", "book", "mean", "p50", "p90", "p99", "p99.9",
                "max");
    for (std::size_t size : sizes) {
        for (const char* scen : {"add", "cancel", "match"}) {
            Bench b(size, 1234 + size);
            std::vector<std::int64_t> lats;
            lats.reserve(n_ops);
            const std::size_t warmup = std::min<std::size_t>(n_ops / 10, 100'000);
            for (std::size_t i = 0; i < warmup + n_ops; ++i) {
                std::int64_t t0 = 0, t1 = 0;
                if (std::strcmp(scen, "add") == 0) {
                    // Timed: add a non-crossing limit. Untimed: cancel a random order.
                    const Side s = (i & 1) ? Side::Buy : Side::Sell;
                    const InputEvent e = b.make_resting(s);
                    b.out.clear();
                    t0 = now_ns();
                    b.eng.process(e, b.out);
                    t1 = now_ns();
                    b.live.add(e.id);
                    b.apply(InputEvent::cancel(b.random_live()));
                    // (the cancelled id is removed from the live set below)
                    b.live.remove(b.out.front().id);
                } else if (std::strcmp(scen, "cancel") == 0) {
                    // Timed: cancel a random resting order. Untimed: add one back.
                    const OrderId id = b.random_live();
                    const InputEvent e = InputEvent::cancel(id);
                    b.out.clear();
                    t0 = now_ns();
                    b.eng.process(e, b.out);
                    t1 = now_ns();
                    b.live.remove(id);
                    b.add_resting((i & 1) ? Side::Buy : Side::Sell);
                } else {
                    // Timed: an aggressive limit that fills exactly one maker at
                    // the touch. Untimed: replace the consumed maker.
                    const Side s = (i & 1) ? Side::Buy : Side::Sell;
                    const Price limit = s == Side::Buy ? kMid + kLevels : kMid - kLevels;
                    const InputEvent e = InputEvent::new_order(b.next_id++, s, OrderType::Limit, limit, kLot);
                    b.out.clear();
                    t0 = now_ns();
                    b.eng.process(e, b.out);
                    t1 = now_ns();
                    for (const OutputEvent& o : b.out)
                        if (o.kind == OutputKind::Trade) b.live.remove(o.maker_id);
                    b.add_resting(opposite(s));
                }
                do_not_optimize(b.out.size());
                if (i >= warmup) lats.push_back(t1 - t0);
            }
            if (b.eng.order_count() != size) {
                std::fprintf(stderr, "book size drifted: %zu != %zu in %s\n", b.eng.order_count(), size, scen);
                return 1;
            }
            Pct p = percentiles(lats);
            std::printf("%-8s %10zu %9.1f %8.0f %8.0f %8.0f %8.0f %10.0f\n", scen, size, p.mean, p.p50, p.p90, p.p99,
                        p.p999, p.max);
            lat << variant << ',' << scen << ',' << size << ',' << n_ops << ',' << p.mean << ',' << p.p50 << ','
                << p.p90 << ',' << p.p99 << ',' << p.p999 << ',' << p.max << '\n';
            write_cdf(cdf, variant, scen, size, lats);
        }
    }

    // Burst throughput per path, no per-op timers. add: 1M non-crossing
    // limits into an empty book. cancel: cancel all of them in random order.
    // match: 1M aggressive orders, each filling exactly one resting maker.
    std::ofstream bt(out_dir + "/burst_" + variant + ".csv");
    bt << "variant,path,orders,wall_s,cpu_s,orders_per_wall_sec,orders_per_cpu_sec\n";
    {
        const std::size_t N = 1'000'000;
        Rng rng(77);
        std::vector<InputEvent> adds, cancels, takers;
        adds.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            const Side s = (i & 1) ? Side::Buy : Side::Sell;
            const Price off = rng.range(1, kLevels);
            adds.push_back(InputEvent::new_order(i + 1, s, OrderType::Limit, s == Side::Buy ? kMid - off : kMid + off,
                                                 kLot));
        }
        for (std::size_t i = 0; i < N; ++i) cancels.push_back(InputEvent::cancel(i + 1));
        for (std::size_t i = N - 1; i > 0; --i)
            std::swap(cancels[i], cancels[static_cast<std::size_t>(rng.range(0, static_cast<std::int64_t>(i)))]);
        for (std::size_t i = 0; i < N; ++i) {
            const Side s = (i & 1) ? Side::Buy : Side::Sell;
            takers.push_back(InputEvent::new_order(N + i + 1, s, OrderType::Limit,
                                                   s == Side::Buy ? kMid + kLevels : kMid - kLevels, kLot));
        }
        auto run = [&](const char* name, MatchingEngine& eng, const std::vector<InputEvent>& evs) {
            std::vector<OutputEvent> out;
            out.reserve(64);
            std::size_t n_out = 0;
            const auto w0 = now_ns();
            const auto c0 = thread_cpu_ns();
            for (const auto& e : evs) {
                out.clear();
                eng.process(e, out);
                n_out += out.size();
            }
            const double ws = static_cast<double>(now_ns() - w0) * 1e-9;
            const double cs = static_cast<double>(thread_cpu_ns() - c0) * 1e-9;
            do_not_optimize(n_out);
            std::printf("burst %-6s %zu orders: %.2f M/s wall, %.2f M/s cpu\n", name, evs.size(),
                        static_cast<double>(evs.size()) / ws / 1e6, static_cast<double>(evs.size()) / cs / 1e6);
            bt << variant << ',' << name << ',' << evs.size() << ',' << ws << ',' << cs << ','
               << static_cast<double>(evs.size()) / ws << ',' << static_cast<double>(evs.size()) / cs << '\n';
        };
        {
            MatchingEngine eng(N + 1024);
            run("add", eng, adds);
            run("cancel", eng, cancels);
        }
        {
            MatchingEngine eng(N + 1024);
            std::vector<OutputEvent> tmp;
            for (const auto& e : adds) {
                tmp.clear();
                eng.process(e, tmp);
            }
            run("match", eng, takers);
            if (eng.order_count() != 0) {
                std::fprintf(stderr, "match burst left %zu orders\n", eng.order_count());
                return 1;
            }
        }
        {
            // Hostile id pattern: ids that share their low 16 bits. A hash that
            // keeps low bits (identity) piles them into a few probe chains.
            const std::size_t M = 20'000;
            std::vector<InputEvent> strided(adds.begin(), adds.begin() + static_cast<std::ptrdiff_t>(M));
            for (std::size_t i = 0; i < M; ++i) strided[i].id = static_cast<OrderId>(i + 1) << 16;
            MatchingEngine eng(M + 1024);
            run("add_strided_ids", eng, strided);
        }
    }

    // Throughput: the random mixed flow, pre-generated, with no per-event timers.
    std::ofstream thr(out_dir + "/throughput_" + variant + ".csv");
    thr << "variant,events,new_orders,outputs,trades,wall_s,cpu_s,events_per_wall_sec,events_per_cpu_sec,new_orders_per_cpu_sec,resting_at_end\n";
    for (int rep = 0; rep < 3; ++rep) {
        FlowConfig c;
        c.seed = 2024 + static_cast<std::uint64_t>(rep);
        c.max_live = 10'000;
        c.half_width = 50;
        const auto flow = OrderFlow(c).take(flow_events);
        std::size_t news = 0;
        for (const auto& e : flow) news += e.kind == InputKind::New;
        MatchingEngine eng(1 << 16);
        std::vector<OutputEvent> out;
        out.reserve(1024);
        std::uint64_t outputs = 0, trades = 0;
        const auto t0 = now_ns();
        const auto c0 = thread_cpu_ns();
        for (const auto& e : flow) {
            out.clear();
            eng.process(e, out);
            outputs += out.size();
            for (const auto& o : out) trades += o.kind == OutputKind::Trade;
        }
        const double s = static_cast<double>(now_ns() - t0) * 1e-9;
        const double cs = static_cast<double>(thread_cpu_ns() - c0) * 1e-9;
        const double n_ev = static_cast<double>(flow.size());
        std::printf("flow rep %d: %zu events, %.2f M events/s wall, %.2f M events/s cpu (%.2f M new orders/s cpu), "
                    "%" PRIu64 " trades\n",
                    rep, flow.size(), n_ev / s / 1e6, n_ev / cs / 1e6, static_cast<double>(news) / cs / 1e6, trades);
        thr << variant << ',' << flow.size() << ',' << news << ',' << outputs << ',' << trades << ',' << s << ','
            << cs << ',' << n_ev / s << ',' << n_ev / cs << ',' << static_cast<double>(news) / cs << ','
            << eng.order_count() << '\n';
    }
    return 0;
}
