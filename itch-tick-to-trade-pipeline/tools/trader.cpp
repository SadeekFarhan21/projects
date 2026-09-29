// trader: runs the optimized pipeline once over a capture and prints a report.
//
//   trader --capture data/capture.itch [--inline] [--no-pace] [--speed X]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "llt/pipeline.hpp"
#include "llt/stats.hpp"

int main(int argc, char** argv) {
    std::string path = "data/capture.itch";
    bool inline_mode = false;
    llt::PipelineConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--capture") && i + 1 < argc) path = argv[++i];
        else if (!std::strcmp(argv[i], "--inline")) inline_mode = true;
        else if (!std::strcmp(argv[i], "--no-pace")) cfg.pace = false;
        else if (!std::strcmp(argv[i], "--speed") && i + 1 < argc) cfg.speed = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--qos") && i + 1 < argc) {
            const std::string q = argv[++i];
            cfg.qos = q == "default" ? llt::Qos::Default
                    : q == "background" ? llt::Qos::Background
                    : q == "fixed" ? llt::Qos::Fixed
                    : q == "realtime" ? llt::Qos::Realtime
                                      : llt::Qos::Interactive;
        }
        else {
            std::fprintf(stderr, "usage: trader --capture FILE [--inline] [--no-pace] [--speed X] [--qos default|interactive|background|fixed|realtime]\n");
            return 2;
        }
    }
    const llt::Capture cap = llt::load_capture(path);
    std::printf("capture %s: %zu messages, %zu bytes\n", path.c_str(), cap.message_count,
                cap.bytes.size());
    std::printf("counter: %llu Hz (%.2f ns/tick)\n", (unsigned long long)llt::clk::freq_hz(),
                llt::clk::ns_per_tick());

    using Book = llt::ArrayBook<llt::FlatOrderMap>;
    const llt::RunResult r = inline_mode ? llt::run_inline<Book>(cap, cfg)
                                         : llt::run_threaded<Book, llt::SpscRing, false>(cap, cfg);

    std::printf("mode=%s pace=%s speed=%.2f qos=%s\n", inline_mode ? "inline" : "threaded",
                cfg.pace ? "on" : "off", cfg.speed, llt::to_string(cfg.qos));
    std::printf("md messages=%zu wall=%.3f ms throughput=%.2f M msg/s\n", r.md_messages,
                r.wall_ns / 1e6, r.md_messages / (r.wall_ns / 1e9) / 1e6);
    std::printf("orders sent=%zu seq_errors=%llu decode_errors=%llu backpressure_spins=%llu\n",
                r.orders, (unsigned long long)r.seq_errors, (unsigned long long)r.decode_errors,
                (unsigned long long)r.backpressure_spins);
    std::printf("risk verdicts:");
    for (std::size_t i = 0; i < r.verdicts.size(); ++i)
        std::printf(" %s=%llu", llt::to_string(static_cast<llt::RiskVerdict>(i)),
                    (unsigned long long)r.verdicts[i]);
    std::printf("\nbook: unknown_ref=%llu out_of_band=%llu table_full=%llu\n",
                (unsigned long long)r.book.unknown_ref, (unsigned long long)r.book.out_of_band,
                (unsigned long long)r.book.table_full);
    std::printf("placement feed: %s\n", llt::describe(r.placement[0]).c_str());
    std::printf("tick-to-trade: %s\n", llt::format_summary(llt::summarize(llt::tick_to_trade_ns(r))).c_str());
    return 0;
}
