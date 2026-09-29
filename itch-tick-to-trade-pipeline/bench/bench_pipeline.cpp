// End-to-end tick-to-trade ablation.
//
// Replays the capture through each pipeline variant, paced at capture
// timestamps, `reps` times each, then once unpaced for throughput. Every
// variant must produce the identical order stream (checked by hash).
//
//   bench_pipeline --capture data/capture.itch [--reps 5] [--speed 1.0] [--out results]
//
// Outputs (in --out):
//   pipeline_runs.csv     one row per (config, rep)
//   pipeline_summary.csv  pooled percentiles, throughput, stage breakdown per config
//   pipeline_hist.csv     tick-to-trade histogram in counter ticks (41.67 ns) per config
//   pipeline_env.txt      counter frequency, thread placement return codes
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "llt/pipeline.hpp"
#include "llt/stats.hpp"

using namespace llt;

namespace {

using Fast = ArrayBook<FlatOrderMap>;
using RunFn = RunResult (*)(const Capture&, const PipelineConfig&);

struct Variant {
    const char* name;
    const char* change;  // what differs from "optimized"
    Qos qos;
    RunFn fn;
};

const Variant kVariants[] = {
    {"optimized", "SPSC padded ring, array book + flat hash, values in slots, P-core QoS",
     Qos::Interactive, &run_threaded<Fast, SpscRing, false>},
    {"mutex_queue", "std::mutex ring instead of SPSC", Qos::Interactive,
     &run_threaded<Fast, MutexQueue, false>},
    {"naive_spsc", "SPSC without padding/cached indices, seq_cst", Qos::Interactive,
     &run_threaded<Fast, SpscNaive, false>},
    {"map_book", "std::map levels + std::unordered_map orders", Qos::Interactive,
     &run_threaded<MapBook, SpscRing, false>},
    {"std_order_table", "array ladder but std::unordered_map orders", Qos::Interactive,
     &run_threaded<ArrayBook<StdOrderMap>, SpscRing, false>},
    {"heap_messages", "new/delete per message, pointers in ring", Qos::Interactive,
     &run_threaded<Fast, SpscRing, true>},
    {"default_qos", "no QoS class set", Qos::Default, &run_threaded<Fast, SpscRing, false>},
    {"background_qos", "QOS_CLASS_BACKGROUND (efficiency cores)", Qos::Background,
     &run_threaded<Fast, SpscRing, false>},
    {"inline_1thread", "feed+book+strategy+risk+gateway on one thread", Qos::Interactive,
     &run_inline<Fast>},
    {"all_naive", "mutex queue + map book + heap messages + default QoS", Qos::Default,
     &run_threaded<MapBook, MutexQueue, true>},
};

uint64_t order_hash(const RunResult& r) {
    uint64_t h = 1469598103934665603ull;  // FNV-1a over the fields the exchange saw
    auto mix = [&](const void* p, std::size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    };
    for (const auto& o : r.exch_orders) {
        mix(&o.token, 4);
        mix(&o.side, 1);
        mix(&o.shares, 4);
        mix(o.stock, 8);
        mix(&o.price, 4);
    }
    return h;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    std::string path = "data/capture.itch", out = "results";
    int reps = 5;
    double speed = 1.0;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--capture") && i + 1 < argc) path = argv[++i];
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--speed") && i + 1 < argc) speed = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else {
            std::fprintf(stderr, "usage: bench_pipeline --capture FILE [--reps N] [--speed X] [--out DIR] [--only NAME]\n");
            return 2;
        }
    }
    const Capture cap = load_capture(path);
    std::printf("capture %s: %zu messages; counter %llu Hz (%.2f ns/tick); reps=%d speed=%.2f\n",
                path.c_str(), cap.message_count, (unsigned long long)clk::freq_hz(), clk::ns_per_tick(),
                reps, speed);

    FILE* runs = std::fopen((out + "/pipeline_runs.csv").c_str(), "w");
    FILE* summ = std::fopen((out + "/pipeline_summary.csv").c_str(), "w");
    FILE* hist = std::fopen((out + "/pipeline_hist.csv").c_str(), "w");
    FILE* env = std::fopen((out + "/pipeline_env.txt").c_str(), "w");
    if (!runs || !summ || !hist || !env) {
        std::perror("open results");
        return 1;
    }
    std::fprintf(runs, "config,rep,orders,mean_ns,p50_ns,p90_ns,p99_ns,p999_ns,max_ns,wall_ms,backpressure_spins\n");
    std::fprintf(summ,
                 "config,change,orders_per_rep,samples,mean_ns,p50_ns,p90_ns,p99_ns,p999_ns,max_ns,"
                 "median_rep_p99_ns,feed_to_decide_p50_ns,decide_to_wire_p50_ns,wire_to_exch_p50_ns,"
                 "unpaced_mmsg_per_s,order_hash\n");
    std::fprintf(hist, "config,ticks,ns,count\n");
    std::fprintf(env, "counter_hz=%llu ns_per_tick=%.4f messages=%zu reps=%d speed=%.2f\n",
                 (unsigned long long)clk::freq_hz(), clk::ns_per_tick(), cap.message_count, reps, speed);

    uint64_t ref_hash = 0;
    bool all_match = true;
    for (const Variant& v : kVariants) {
        if (!only.empty() && only != v.name) continue;
        PipelineConfig cfg;
        cfg.qos = v.qos;
        cfg.speed = speed;
        std::vector<double> pooled, d1, d2, d3, rep_p99;
        std::size_t orders = 0;
        uint64_t h = 0;
        for (int rep = 0; rep < reps; ++rep) {
            const RunResult r = v.fn(cap, cfg);
            auto ns = tick_to_trade_ns(r);
            const auto s = summarize(ns);
            std::fprintf(runs, "%s,%d,%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%llu\n", v.name, rep, r.orders,
                         s.mean, s.p50, s.p90, s.p99, s.p999, s.max, r.wall_ns / 1e6,
                         (unsigned long long)r.backpressure_spins);
            std::fflush(runs);
            rep_p99.push_back(s.p99);
            pooled.insert(pooled.end(), ns.begin(), ns.end());
            for (const auto& t : r.timings) {
                d1.push_back(clk::to_ns(t.t_decide - t.t_in));
                d2.push_back(clk::to_ns(t.t_out - t.t_decide));
                d3.push_back(clk::to_ns(t.t_rx - t.t_out));
            }
            orders = r.orders;
            h = order_hash(r);
            if (rep == 0) {
                std::fprintf(env, "%s placement[0]: %s\n", v.name, describe(r.placement[0]).c_str());
                if (r.book.unknown_ref || r.book.out_of_band || r.seq_errors)
                    std::fprintf(stderr, "WARNING %s: unknown_ref=%llu oob=%llu seq_errors=%llu\n", v.name,
                                 (unsigned long long)r.book.unknown_ref, (unsigned long long)r.book.out_of_band,
                                 (unsigned long long)r.seq_errors);
            }
        }
        if (ref_hash == 0) ref_hash = h;
        all_match &= (h == ref_hash);

        PipelineConfig fast = cfg;
        fast.pace = false;
        const RunResult tr = v.fn(cap, fast);
        const double mps = tr.md_messages / (tr.wall_ns / 1e9) / 1e6;

        const auto s = summarize(pooled);
        std::fprintf(summ, "%s,\"%s\",%zu,%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,%016llx\n",
                     v.name, v.change, orders, s.n, s.mean, s.p50, s.p90, s.p99, s.p999, s.max,
                     median_of(rep_p99), median_of(d1), median_of(d2), median_of(d3), mps,
                     (unsigned long long)h);
        std::fflush(summ);

        std::map<uint64_t, std::size_t> hticks;
        for (double x : pooled) ++hticks[static_cast<uint64_t>(x / clk::ns_per_tick() + 0.5)];
        for (auto& [t, c] : hticks) std::fprintf(hist, "%s,%llu,%.2f,%zu\n", v.name, (unsigned long long)t,
                                                 t * clk::ns_per_tick(), c);
        std::fflush(hist);

        std::printf("%-16s orders/rep=%-6zu p50=%7.0f p90=%7.0f p99=%8.0f p99.9=%9.0f max=%10.0f ns | "
                    "unpaced %.2f M msg/s | hash %016llx\n",
                    v.name, orders, s.p50, s.p90, s.p99, s.p999, s.max, mps, (unsigned long long)h);
    }
    std::fprintf(env, "all_order_streams_identical=%s\n", all_match ? "yes" : "no");
    std::printf("all order streams identical: %s\n", all_match ? "yes" : "NO");
    std::fclose(runs);
    std::fclose(summ);
    std::fclose(hist);
    std::fclose(env);
    return all_match ? 0 : 1;
}
