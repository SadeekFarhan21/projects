// exch: command-line front end.
//
//   exch gen    --n N --seed S --out inputs.log         write a random input log
//   exch replay --in inputs.log [--out outputs.log] [--l2 l2.csv] [--trades trades.csv] [--depth K]
//   exch verify --in inputs.log (--expect outputs.log | --digest HEX)
//   exch fuzz   --n N [--seed S] [--seeds K] [--check-every M] [--width W] [--max-live L]
//
// replay is the deterministic replay tool: the same input log always yields
// the same output stream and digest, on any machine.
#include "exchange/engine.hpp"
#include "exchange/market_data.hpp"
#include "exchange/order_flow.hpp"
#include "exchange/reference.hpp"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace exch;

namespace {

struct Args {
    std::map<std::string, std::string> kv;
    bool has(const std::string& k) const { return kv.count(k) != 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    std::uint64_t num(const std::string& k, std::uint64_t def) const {
        return has(k) ? std::strtoull(get(k).c_str(), nullptr, 10) : def;
    }
};

Args parse(int argc, char** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.kv[k] = argv[++i];
        else a.kv[k] = "1";
    }
    return a;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool load_inputs(const std::string& path, std::vector<InputEvent>& v) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    std::string line;
    std::size_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        InputEvent e;
        if (!decode(line, e)) {
            std::fprintf(stderr, "%s:%zu: cannot parse '%s'\n", path.c_str(), lineno, line.c_str());
            return false;
        }
        v.push_back(e);
    }
    return true;
}

int cmd_gen(const Args& a) {
    FlowConfig c;
    c.seed = a.num("seed", 1);
    const std::uint64_t n = a.num("n", 100000);
    const std::string out = a.get("out", "inputs.log");
    std::ofstream f(out);
    f << "# exch input log, generated with seed " << c.seed << ", " << n << " events\n";
    OrderFlow flow(c);
    for (std::uint64_t i = 0; i < n; ++i) f << encode(flow.next()) << '\n';
    std::printf("wrote %" PRIu64 " input events to %s\n", n, out.c_str());
    return 0;
}

int cmd_replay(const Args& a) {
    std::vector<InputEvent> in;
    if (!load_inputs(a.get("in", "inputs.log"), in)) return 2;

    std::ofstream out_f, l2_f, tr_f;
    if (a.has("out")) out_f.open(a.get("out"));
    if (a.has("l2")) { l2_f.open(a.get("l2")); l2_f << "seq,side,price,qty\n"; }
    if (a.has("trades")) { tr_f.open(a.get("trades")); tr_f << "seq,trade_id,aggressor,price,qty\n"; }

    MatchingEngine eng(1 << 16);
    MarketDataFeed md;
    md.keep_tape = false;
    if (l2_f.is_open())
        md.on_l2 = [&](const OutputEvent& e) {
            l2_f << e.seq << ',' << to_string(e.side) << ',' << e.price << ',' << e.qty << '\n';
        };
    if (tr_f.is_open())
        md.on_trade = [&](const TradePrint& t) {
            tr_f << t.seq << ',' << t.id << ',' << to_string(t.aggressor) << ',' << t.price << ',' << t.qty << '\n';
        };
    StreamDigest dig;
    std::vector<OutputEvent> buf;
    std::uint64_t trades = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const InputEvent& e : in) {
        buf.clear();
        eng.process(e, buf);
        for (const OutputEvent& o : buf) {
            dig.add(o);
            md.on_event(o);
            if (o.kind == OutputKind::Trade) ++trades;
            if (out_f.is_open()) out_f << encode(o) << '\n';
        }
    }
    const double secs = seconds_since(t0);
    std::printf("inputs        %zu\n", in.size());
    std::printf("outputs       %" PRIu64 "\n", dig.count());
    std::printf("trades        %" PRIu64 " (volume %" PRId64 ")\n", trades, md.traded_volume());
    std::printf("l2 updates    %" PRIu64 "\n", md.book_updates());
    std::printf("resting       %zu orders, %zu bid levels, %zu ask levels\n", eng.order_count(),
                eng.level_count(Side::Buy), eng.level_count(Side::Sell));
    std::printf("digest        %016" PRIx64 "\n", dig.value());
    std::printf("elapsed       %.3f s (including I/O)\n", secs);
    const std::size_t k = a.num("depth", 5);
    const Depth bids = eng.depth(Side::Buy, k), asks = eng.depth(Side::Sell, k);
    std::printf("%12s %8s | %-8s %-12s\n", "bid qty", "bid", "ask", "ask qty");
    for (std::size_t i = 0; i < std::max(bids.size(), asks.size()); ++i) {
        char l[64] = "", r[64] = "";
        if (i < bids.size()) std::snprintf(l, sizeof l, "%12" PRId64 " %8" PRId64, bids[i].second, bids[i].first);
        else std::snprintf(l, sizeof l, "%21s", "");
        if (i < asks.size()) std::snprintf(r, sizeof r, "%-8" PRId64 " %-12" PRId64, asks[i].first, asks[i].second);
        std::printf("%s | %s\n", l, r);
    }
    return 0;
}

int cmd_verify(const Args& a) {
    std::vector<InputEvent> in;
    if (!load_inputs(a.get("in", "inputs.log"), in)) return 2;
    MatchingEngine eng;
    StreamDigest dig;
    std::vector<OutputEvent> buf;
    std::ifstream expect;
    if (a.has("expect")) expect.open(a.get("expect"));
    std::string line;
    std::uint64_t n = 0;
    for (const InputEvent& e : in) {
        buf.clear();
        eng.process(e, buf);
        for (const OutputEvent& o : buf) {
            dig.add(o);
            ++n;
            if (expect.is_open()) {
                if (!std::getline(expect, line)) {
                    std::printf("MISMATCH: expected log ended before output %" PRIu64 "\n", n);
                    return 1;
                }
                const std::string got = encode(o);
                if (got != line) {
                    std::printf("MISMATCH at output %" PRIu64 "\n  expected: %s\n  got:      %s\n", n, line.c_str(),
                                got.c_str());
                    return 1;
                }
            }
        }
    }
    if (expect.is_open() && std::getline(expect, line)) {
        std::printf("MISMATCH: expected log has extra lines after %" PRIu64 " outputs\n", n);
        return 1;
    }
    if (a.has("digest")) {
        const std::uint64_t want = std::strtoull(a.get("digest").c_str(), nullptr, 16);
        if (want != dig.value()) {
            std::printf("MISMATCH: digest %016" PRIx64 " != expected %016" PRIx64 "\n", dig.value(), want);
            return 1;
        }
    }
    std::printf("OK %" PRIu64 " outputs, digest %016" PRIx64 "\n", n, dig.value());
    return 0;
}

// Differential fuzzing against the reference matcher. Exits non-zero on the
// first divergence and prints the event where it happened.
int cmd_fuzz(const Args& a) {
    const std::uint64_t n = a.num("n", 1'000'000);
    const std::uint64_t seed0 = a.num("seed", 1);
    const std::uint64_t seeds = a.num("seeds", 1);
    const std::uint64_t check_every = a.num("check-every", 1000);
    std::uint64_t total_events = 0, total_outputs = 0, total_trades = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint64_t s = seed0; s < seed0 + seeds; ++s) {
        FlowConfig c;
        c.seed = s;
        // Vary the shape of the flow across seeds.
        c.half_width = a.has("width") ? static_cast<Price>(a.num("width", 20)) : 5 + static_cast<Price>(s % 4) * 15;
        c.max_qty = (s % 3 == 0) ? 10 : 100;
        if (a.has("max-live")) {
            c.max_live = a.num("max-live", 500);
            c.p_cancel = 0.15; // let the book actually fill up to the cap
        }
        OrderFlow flow(c);
        MatchingEngine eng;
        ReferenceMatcher ref;
        MarketDataFeed md;
        md.keep_tape = false;
        std::vector<OutputEvent> a_out, b_out;
        for (std::uint64_t i = 0; i < n; ++i) {
            const InputEvent e = flow.next();
            a_out.clear();
            b_out.clear();
            eng.process(e, a_out);
            ref.process(e, b_out);
            if (a_out != b_out) {
                std::printf("DIVERGENCE seed %" PRIu64 " event %" PRIu64 ": %s\n", s, i, encode(e).c_str());
                std::printf("engine:\n");
                for (auto& o : a_out) std::printf("  %s\n", encode(o).c_str());
                std::printf("reference:\n");
                for (auto& o : b_out) std::printf("  %s\n", encode(o).c_str());
                return 1;
            }
            for (auto& o : a_out) {
                md.on_event(o);
                total_trades += o.kind == OutputKind::Trade;
            }
            total_outputs += a_out.size();
            if (i % check_every == 0) {
                eng.check_invariants();
                if (eng.orders_in_priority(Side::Buy) != ref.orders_in_priority(Side::Buy) ||
                    eng.orders_in_priority(Side::Sell) != ref.orders_in_priority(Side::Sell) ||
                    md.depth(Side::Buy) != eng.depth(Side::Buy) || md.depth(Side::Sell) != eng.depth(Side::Sell)) {
                    std::printf("STATE DIVERGENCE seed %" PRIu64 " event %" PRIu64 "\n", s, i);
                    return 1;
                }
            }
        }
        total_events += n;
        std::printf("seed %" PRIu64 ": %" PRIu64 " events OK (resting at end: %zu)\n", s, n, eng.order_count());
    }
    std::printf("fuzz OK: %" PRIu64 " events, %" PRIu64 " outputs, %" PRIu64 " trades, %.1f s\n", total_events,
                total_outputs, total_trades, seconds_since(t0));
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: exch <gen|replay|verify|fuzz> [options]\n"
                 "  gen    --n N --seed S --out FILE\n"
                 "  replay --in FILE [--out FILE] [--l2 FILE.csv] [--trades FILE.csv] [--depth K]\n"
                 "  verify --in FILE (--expect OUTPUTS | --digest HEX)\n"
                 "  fuzz   --n N [--seed S] [--seeds K] [--check-every M] [--width W] [--max-live L]\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    const Args a = parse(argc, argv, 2);
    if (cmd == "gen") return cmd_gen(a);
    if (cmd == "replay") return cmd_replay(a);
    if (cmd == "verify") return cmd_verify(a);
    if (cmd == "fuzz") return cmd_fuzz(a);
    usage();
    return 2;
}
