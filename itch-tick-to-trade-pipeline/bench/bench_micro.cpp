// Component microbenchmarks. Hand-rolled harness: every measurement times a
// large batch with steady_clock and divides, so the 41.67 ns counter tick
// does not limit precision. Writes CSVs into --out (default results/).
//
//   bench_micro [--out results] [--messages 1000000]
#include <mach/mach_time.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "llt/capture.hpp"
#include "llt/clock.hpp"
#include "llt/itch.hpp"
#include "llt/order_book.hpp"
#include "llt/pool.hpp"
#include "llt/queues.hpp"
#include "llt/stats.hpp"
#include "llt/synth.hpp"
#include "llt/thread_util.hpp"

using namespace llt;
using SteadyClock = std::chrono::steady_clock;

namespace {

template <class T>
inline void do_not_optimize(const T& v) {
    asm volatile("" : : "r,m"(v) : "memory");
}

double since_ns(SteadyClock::time_point t0) {
    return std::chrono::duration<double, std::nano>(SteadyClock::now() - t0).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------------------------------------------------------- clocks
template <class F>
double clock_cost_ns(F f, int n) {
    uint64_t sink = 0;
    const auto t0 = SteadyClock::now();
    for (int i = 0; i < n; ++i) sink += static_cast<uint64_t>(f());
    const double ns = since_ns(t0);
    do_not_optimize(sink);
    return ns / n;
}

template <class F>
uint64_t clock_min_step(F f) {
    // Smallest non-zero difference between back-to-back reads.
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < 200'000; ++i) {
        const uint64_t a = static_cast<uint64_t>(f());
        uint64_t b;
        do b = static_cast<uint64_t>(f());
        while (b == a);
        best = std::min(best, b - a);
    }
    return best;
}

void bench_clocks(FILE* out) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    const int n = 20'000'000;
    const double tick_ns = clk::ns_per_tick();
    struct Row {
        const char* name;
        double cost;
        double step_ns;
    };
    std::vector<Row> rows;
    rows.push_back({"cntvct_el0+isb (clk::now)", clock_cost_ns([] { return clk::now(); }, n),
                    clock_min_step([] { return clk::now(); }) * tick_ns});
    rows.push_back({"cntvct_el0 no barrier", clock_cost_ns([] { return clk::now_unordered(); }, n),
                    clock_min_step([] { return clk::now_unordered(); }) * tick_ns});
    rows.push_back({"mach_absolute_time", clock_cost_ns([] { return mach_absolute_time(); }, n),
                    clock_min_step([] { return mach_absolute_time(); }) * tb.numer / double(tb.denom)});
    rows.push_back({"steady_clock::now",
                    clock_cost_ns([] { return SteadyClock::now().time_since_epoch().count(); }, n),
                    double(clock_min_step([] { return SteadyClock::now().time_since_epoch().count(); }))});
    std::fprintf(out, "clock,cost_ns_per_call,min_observed_step_ns\n");
    std::printf("\n== clocks (counter freq %llu Hz, mach timebase %u/%u) ==\n",
                (unsigned long long)clk::freq_hz(), tb.numer, tb.denom);
    for (auto& r : rows) {
        std::fprintf(out, "%s,%.2f,%.2f\n", r.name, r.cost, r.step_ns);
        std::printf("  %-28s %6.2f ns/call   min step %.2f ns\n", r.name, r.cost, r.step_ns);
    }
}

// ---------------------------------------------------------------- queues
template <template <class, std::size_t> class Q>
void queue_bench(const char* name, FILE* out, int reps) {
    constexpr int kRoundTrips = 1'000'000;
    constexpr uint64_t kItems = 20'000'000;
    std::vector<double> rtt, per_sample_p50, per_sample_p99, tput;
    std::vector<double> samples(kRoundTrips);
    for (int rep = 0; rep < reps; ++rep) {
        {  // ping-pong round trip
            auto ab = std::make_unique<Q<uint64_t, 1024>>();
            auto ba = std::make_unique<Q<uint64_t, 1024>>();
            std::thread echo([&] {
                place_current_thread(Qos::Interactive, 1);
                uint64_t v;
                for (int i = 0; i < kRoundTrips; ++i) {
                    pop_spin(*ab, v);
                    push_spin(*ba, v);
                }
            });
            place_current_thread(Qos::Interactive, 0);
            uint64_t v;
            const auto t0 = SteadyClock::now();
            for (int i = 0; i < kRoundTrips; ++i) {
                const uint64_t s = clk::now();
                push_spin(*ab, static_cast<uint64_t>(i));
                pop_spin(*ba, v);
                samples[i] = clk::to_ns(clk::now() - s);
            }
            const double total = since_ns(t0);
            echo.join();
            rtt.push_back(total / kRoundTrips);
            auto s = summarize(samples);
            per_sample_p50.push_back(s.p50);
            per_sample_p99.push_back(s.p99);
        }
        {  // one-way throughput
            auto q = std::make_unique<Q<uint64_t, 1024>>();
            std::thread prod([&] {
                place_current_thread(Qos::Interactive, 1);
                for (uint64_t i = 0; i < kItems; ++i) push_spin(*q, i);
            });
            uint64_t v, sum = 0;
            const auto t0 = SteadyClock::now();
            for (uint64_t i = 0; i < kItems; ++i) {
                pop_spin(*q, v);
                sum += v;
            }
            const double ns = since_ns(t0);
            prod.join();
            do_not_optimize(sum);
            tput.push_back(kItems / (ns / 1e9) / 1e6);
        }
    }
    const double m_rtt = median(rtt);
    std::fprintf(out, "%s,%.1f,%.1f,%.0f,%.0f,%.1f\n", name, m_rtt, m_rtt / 2, median(per_sample_p50),
                 median(per_sample_p99), median(tput));
    std::printf("  %-12s rtt %7.1f ns (one-way ~%6.1f)  sample p50 %5.0f p99 %6.0f ns  tput %7.1f M/s\n",
                name, m_rtt, m_rtt / 2, median(per_sample_p50), median(per_sample_p99), median(tput));
}

// ---------------------------------------------------------------- books
std::vector<MdEvent> decode_all(const Capture& c) {
    std::vector<MdEvent> evs;
    evs.reserve(c.message_count);
    itch::FrameCursor cur{c.bytes.data(), c.bytes.data() + c.bytes.size()};
    const uint8_t* m;
    uint16_t len;
    while (cur.next(m, len)) {
        MdEvent e;
        if (itch::decode(m, len, e) && e.type != 'R') evs.push_back(e);
    }
    return evs;
}

template <class Book>
void book_bench(const char* name, const std::vector<MdEvent>& evs, FILE* out, int reps) {
    std::vector<double> per;
    uint64_t check = 0;
    for (int rep = 0; rep < reps; ++rep) {
        auto b = std::make_unique<Book>(100, 4096, 1u << 21);
        const auto t0 = SteadyClock::now();
        for (const MdEvent& e : evs) {
            b->apply(e);
            const Top t = b->top(e.locate);
            check += t.bid_px ^ t.ask_qty;
        }
        per.push_back(since_ns(t0) / evs.size());
    }
    do_not_optimize(check);
    std::fprintf(out, "%s,%zu,%.2f,%.2f,%.2f\n", name, evs.size(), median(per),
                 *std::min_element(per.begin(), per.end()), *std::max_element(per.begin(), per.end()));
    std::printf("  %-28s %6.2f ns/event (min %.2f, max %.2f)\n", name, median(per),
                *std::min_element(per.begin(), per.end()), *std::max_element(per.begin(), per.end()));
}

// ---------------------------------------------------------------- allocation
void alloc_bench(FILE* out, int reps) {
    constexpr int kN = 10'000'000;
    std::vector<double> heap, pool, xheap, xvalue;
    for (int rep = 0; rep < reps; ++rep) {
        {
            const auto t0 = SteadyClock::now();
            for (int i = 0; i < kN; ++i) {
                MdEvent* e = new MdEvent;
                e->qty = static_cast<uint32_t>(i);
                do_not_optimize(e);
                delete e;
            }
            heap.push_back(since_ns(t0) / kN);
        }
        {
            ObjectPool<MdEvent> p(1024);
            const auto t0 = SteadyClock::now();
            for (int i = 0; i < kN; ++i) {
                MdEvent* e = p.acquire();
                e->qty = static_cast<uint32_t>(i);
                do_not_optimize(e);
                p.release(e);
            }
            pool.push_back(since_ns(t0) / kN);
        }
        {  // producer allocates, consumer frees on another core
            auto q = std::make_unique<SpscRing<MdEvent*, 1u << 14>>();
            std::thread prod([&] {
                place_current_thread(Qos::Interactive, 1);
                for (int i = 0; i < kN; ++i) {
                    MdEvent* e = new MdEvent;
                    e->qty = static_cast<uint32_t>(i);
                    push_spin(*q, e);
                }
            });
            const auto t0 = SteadyClock::now();
            MdEvent* e;
            uint64_t sum = 0;
            for (int i = 0; i < kN; ++i) {
                pop_spin(*q, e);
                sum += e->qty;
                delete e;
            }
            xheap.push_back(since_ns(t0) / kN);
            prod.join();
            do_not_optimize(sum);
        }
        {  // same transfer, value in the ring slot
            auto q = std::make_unique<SpscRing<MdEvent, 1u << 14>>();
            std::thread prod([&] {
                place_current_thread(Qos::Interactive, 1);
                MdEvent e;
                for (int i = 0; i < kN; ++i) {
                    e.qty = static_cast<uint32_t>(i);
                    push_spin(*q, e);
                }
            });
            const auto t0 = SteadyClock::now();
            MdEvent e;
            uint64_t sum = 0;
            for (int i = 0; i < kN; ++i) {
                pop_spin(*q, e);
                sum += e.qty;
            }
            xvalue.push_back(since_ns(t0) / kN);
            prod.join();
            do_not_optimize(sum);
        }
    }
    std::fprintf(out, "variant,ns_per_message\n");
    std::fprintf(out, "new+delete same thread,%.2f\n", median(heap));
    std::fprintf(out, "pool acquire+release same thread,%.2f\n", median(pool));
    std::fprintf(out, "cross-thread new/delete via ring,%.2f\n", median(xheap));
    std::fprintf(out, "cross-thread by value in ring,%.2f\n", median(xvalue));
    std::printf("  new+delete same thread          %6.2f ns\n", median(heap));
    std::printf("  pool acquire+release            %6.2f ns\n", median(pool));
    std::printf("  cross-thread new/delete via ring %6.2f ns/msg\n", median(xheap));
    std::printf("  cross-thread value in ring       %6.2f ns/msg\n", median(xvalue));
}

FILE* open_out(const std::string& dir, const char* name) {
    const std::string p = dir + "/" + name;
    FILE* f = std::fopen(p.c_str(), "w");
    if (!f) {
        std::perror(p.c_str());
        std::exit(1);
    }
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_dir = "results";
    uint64_t messages = 1'000'000;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--messages") && i + 1 < argc) messages = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    place_current_thread(Qos::Interactive, -1);

    FILE* f = open_out(out_dir, "micro_clock.csv");
    bench_clocks(f);
    std::fclose(f);

    std::printf("\n== queues: 1M ping-pong round trips + 20M one-way items, median of %d ==\n", reps);
    f = open_out(out_dir, "micro_queue.csv");
    std::fprintf(f, "queue,rtt_ns,one_way_ns,sample_rtt_p50_ns,sample_rtt_p99_ns,throughput_mops\n");
    queue_bench<SpscRing>("spsc_padded", f, reps);
    queue_bench<SpscNaive>("spsc_naive", f, reps);
    queue_bench<MutexQueue>("mutex", f, reps);
    std::fclose(f);

    SynthParams sp;
    sp.messages = messages;
    const Capture cap = capture_from_bytes(generate_capture(sp));
    const auto evs = decode_all(cap);
    std::printf("\n== books: apply + top over %zu events, median of %d ==\n", evs.size(), reps);
    f = open_out(out_dir, "micro_book.csv");
    std::fprintf(f, "book,events,ns_per_event_median,ns_min,ns_max\n");
    book_bench<MapBook>("std::map + unordered_map", evs, f, reps);
    book_bench<ArrayBook<StdOrderMap>>("array ladder + unordered_map", evs, f, reps);
    book_bench<ArrayBook<FlatOrderMap>>("array ladder + flat hash", evs, f, reps);
    std::fclose(f);

    std::printf("\n== allocation: 10M messages, median of %d ==\n", reps);
    f = open_out(out_dir, "micro_alloc.csv");
    alloc_bench(f, reps);
    std::fclose(f);
    return 0;
}
