// Query latency and throughput benchmark.
//
//   se_bench --index f.idx --queries queries.jsonl
//            [--mode bm25|taat|and|phrase|codec] [--k 10] [--reps 5]
//            [--threads 1] [--limit N] [--json out.json] [--csv latencies.csv]
//
// Modes
//   bm25    disjunctive BM25, document-at-a-time with a top-k heap
//   taat    disjunctive BM25, term-at-a-time with a dense accumulator
//   and     boolean AND of the query's words, ranked by BM25
//   phrase  the first two words of each query as a quoted phrase
//   codec   decode every postings list once and compare against scanning
//           the same postings stored as plain uint32 (no queries involved)
//
// Latency: one warm-up pass, then `reps` passes over the query set on one
// thread; every query execution is timed with steady_clock (wall) and with
// CLOCK_THREAD_CPUTIME_ID (CPU), since the machine may be shared.
// Throughput (--threads > 1): each thread runs its own Searcher over a
// disjoint slice of the repeated query stream; QPS = queries / wall time.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <thread>

#include "cli.hpp"
#include "se/corpus.hpp"
#include "se/searcher.hpp"

namespace {

std::string first_two_words(const std::string& q) {
  std::vector<std::string> w;
  std::string cur;
  for (char c : q) {
    bool keep = std::isalnum(static_cast<unsigned char>(c)) || static_cast<unsigned char>(c) >= 0x80;
    if (keep) cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    else if (!cur.empty()) { w.push_back(cur); cur.clear(); }
  }
  if (!cur.empty()) w.push_back(cur);
  if (w.size() < 2) return w.empty() ? "" : w[0];
  return "\"" + w[0] + " " + w[1] + "\"";
}

std::string and_of_words(const std::string& q) {
  std::string out, cur;
  for (char c : q) {
    bool keep = std::isalnum(static_cast<unsigned char>(c)) || static_cast<unsigned char>(c) >= 0x80;
    // lowercased so a literal "OR" in the query text is not read as an operator
    if (keep) cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    else if (!cur.empty()) { out += (out.empty() ? "" : " AND ") + cur; cur.clear(); }
  }
  if (!cur.empty()) out += (out.empty() ? "" : " AND ") + cur;
  return out;
}

double pct(std::vector<double>& v, double p) {
  if (v.empty()) return 0;
  size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size()))) ;
  idx = std::min(v.size() - 1, idx == 0 ? 0 : idx - 1);  // nearest-rank
  std::nth_element(v.begin(), v.begin() + static_cast<long>(idx), v.end());
  return v[idx];
}

int codec_bench(const se::Index& ix, const cli::Args& a) {
  // Plain uint32 copy of every (doc, tf) pair, laid out like the doc streams.
  std::vector<uint32_t> raw;
  raw.reserve(ix.num_postings() * 2);
  for (uint32_t t = 0; t < ix.num_terms(); ++t)
    for (auto c = ix.cursor(ix.term_entry(t)); !c.at_end(); c.next()) {
      raw.push_back(c.doc());
      raw.push_back(c.tf());
    }
  const int reps = static_cast<int>(a.num("reps", 5));
  std::vector<double> vb, rw;
  uint64_t sink = 0;
  for (int r = 0; r < reps + 1; ++r) {
    double t0 = cli::now_s();
    for (uint32_t t = 0; t < ix.num_terms(); ++t)
      for (auto c = ix.cursor(ix.term_entry(t)); !c.at_end(); c.next()) sink += c.doc() ^ c.tf();
    double t1 = cli::now_s();
    for (size_t i = 0; i < raw.size(); i += 2) sink += raw[i] ^ raw[i + 1];
    double t2 = cli::now_s();
    if (r > 0) { vb.push_back(t1 - t0); rw.push_back(t2 - t1); }
  }
  std::sort(vb.begin(), vb.end());
  std::sort(rw.begin(), rw.end());
  const double n = static_cast<double>(ix.num_postings());
  const double vb_ns = vb[vb.size() / 2] / n * 1e9, rw_ns = rw[rw.size() / 2] / n * 1e9;
  std::printf("postings=%.0f vbyte decode %.3f ns/posting, raw uint32 scan %.3f ns/posting (sink %llu)\n",
              n, vb_ns, rw_ns, (unsigned long long)(sink & 1));
  if (a.has("json")) {
    FILE* f = std::fopen(a.get("json").c_str(), "w");
    std::fprintf(f,
                 "{\n  \"mode\": \"codec\",\n  \"index\": \"%s\",\n  \"num_postings\": %.0f,\n"
                 "  \"reps\": %d,\n  \"vbyte_ns_per_posting_median\": %.4f,\n"
                 "  \"raw_uint32_ns_per_posting_median\": %.4f,\n  \"vbyte_doc_stream_bytes_per_posting\": %.4f,\n"
                 "  \"raw_bytes_per_posting\": 8\n}\n",
                 cli::json_escape(a.get("index")).c_str(), n, reps, vb_ns, rw_ns,
                 [&] {
                   uint64_t b = 0;
                   for (uint32_t t = 0; t < ix.num_terms(); ++t) b += ix.term_entry(t).docs_len;
                   return static_cast<double>(b) / n;
                 }());
    std::fclose(f);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  auto a = cli::parse(argc, argv);
  if (a.get("index").empty() || (a.get("queries").empty() && a.get("mode") != "codec"))
    cli::die("usage: se_bench --index f.idx --queries q.jsonl [--mode bm25|taat|and|phrase|codec] ...");
  try {
    auto ix = se::Index::open(a.get("index"));
    const std::string mode = a.get("mode", "bm25");
    if (mode == "codec") return codec_bench(ix, a);

    auto qs = se::load_queries_jsonl(a.get("queries"));
    size_t limit = static_cast<size_t>(a.num("limit", 0));
    if (limit && qs.size() > limit) qs.resize(limit);
    std::vector<std::string> texts;
    for (auto& q : qs) {
      if (mode == "phrase") texts.push_back(first_two_words(q.text));
      else if (mode == "and") texts.push_back(and_of_words(q.text));
      else texts.push_back(q.text);
    }
    const size_t k = static_cast<size_t>(a.num("k", 10));
    const int reps = static_cast<int>(a.num("reps", 5));
    const unsigned threads = static_cast<unsigned>(a.num("threads", 1));

    auto run_one = [&](se::Searcher& s, const std::string& q, size_t* matches) -> size_t {
      if (mode == "bm25") return s.search_bm25(q, k).size();
      if (mode == "taat") return s.search_bm25_taat(q, k).size();
      if (mode == "and" || mode == "phrase") return s.search_boolean(q, k, matches).size();
      cli::die("unknown mode " + mode);
    };

    se::Searcher s(ix);
    // Work per query: total postings of its distinct terms (what an exhaustive
    // disjunctive evaluation must decode), to normalise latency per posting.
    double postings_per_query = 0;
    for (auto& q : qs)
      for (auto& wt : s.weigh_terms(s.tokenizer().terms(q.text)))
        if (wt.entry) postings_per_query += wt.entry->df;
    postings_per_query /= static_cast<double>(std::max<size_t>(1, qs.size()));
    size_t sink = 0, total_matches = 0;
    for (auto& q : texts) sink += run_one(s, q, &total_matches);  // warm-up
    total_matches = 0;
    const double load0 = cli::load_avg_1m();
    std::vector<double> lat, cpu;  // microseconds: wall clock and thread CPU time
    std::vector<double> first_wall(texts.size()), first_cpu(texts.size());
    lat.reserve(texts.size() * static_cast<size_t>(reps));
    cpu.reserve(texts.size() * static_cast<size_t>(reps));
    for (int r = 0; r < reps; ++r)
      for (size_t i = 0; i < texts.size(); ++i) {
        size_t m = 0;
        double c0 = cli::thread_cpu_s();
        double t0 = cli::now_s();
        sink += run_one(s, texts[i], &m);
        double us = (cli::now_s() - t0) * 1e6;
        double cus = (cli::thread_cpu_s() - c0) * 1e6;
        lat.push_back(us);
        cpu.push_back(cus);
        if (r == 0) { first_wall[i] = us; first_cpu[i] = cus; total_matches += m; }
      }
    auto mean_of = [](const std::vector<double>& v) {
      return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };
    const double mean = mean_of(lat), cmean = mean_of(cpu);
    double p50 = pct(lat, 50), p90 = pct(lat, 90), p99 = pct(lat, 99), p999 = pct(lat, 99.9);
    double pmax = *std::max_element(lat.begin(), lat.end());
    double c50 = pct(cpu, 50), c90 = pct(cpu, 90), c99 = pct(cpu, 99), c999 = pct(cpu, 99.9);
    double cmax = *std::max_element(cpu.begin(), cpu.end());

    double qps_mt = 0;
    if (threads > 1) {
      const size_t total = texts.size() * static_cast<size_t>(reps);
      std::atomic<size_t> sink_mt{0};
      double t0 = cli::now_s();
      std::vector<std::thread> pool;
      for (unsigned w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
          se::Searcher ls(ix);
          size_t local = 0;
          for (size_t i = w; i < total; i += threads) local += run_one(ls, texts[i % texts.size()], nullptr);
          sink_mt += local;
        });
      for (auto& th : pool) th.join();
      qps_mt = static_cast<double>(total) / (cli::now_s() - t0);
    }

    std::printf("mode=%s queries=%zu reps=%d k=%zu load=%.0f\n", mode.c_str(), texts.size(), reps, k, load0);
    std::printf("  wall us: mean=%.1f p50=%.1f p90=%.1f p99=%.1f p99.9=%.1f max=%.1f  qps(1t)=%.0f\n", mean,
                p50, p90, p99, p999, pmax, 1e6 / mean);
    std::printf("  cpu  us: mean=%.1f p50=%.1f p90=%.1f p99=%.1f p99.9=%.1f max=%.1f  qps(1t, cpu)=%.0f\n",
                cmean, c50, c90, c99, c999, cmax, 1e6 / cmean);
    std::printf("  postings/query=%.0f  cpu ns/posting=%.2f\n", postings_per_query,
                cmean * 1e3 / std::max(1.0, postings_per_query));
    if (threads > 1) std::printf("  qps(%u threads, wall)=%.0f\n", threads, qps_mt);
    if (mode == "and" || mode == "phrase")
      std::printf("  mean_matches=%.1f\n", static_cast<double>(total_matches) / static_cast<double>(texts.size()));
    if (sink == 42) std::printf("\n");  // keep results observable

    if (a.has("json")) {
      FILE* f = std::fopen(a.get("json").c_str(), "w");
      if (!f) cli::die("cannot write json");
      std::fprintf(f,
                   "{\n  \"mode\": \"%s\",\n  \"index\": \"%s\",\n  \"num_docs\": %u,\n  \"queries\": %zu,\n"
                   "  \"reps\": %d,\n  \"k\": %zu,\n  \"load_avg_1m\": %.2f,\n"
                   "  \"wall_mean_us\": %.3f,\n  \"wall_p50_us\": %.3f,\n  \"wall_p90_us\": %.3f,\n"
                   "  \"wall_p99_us\": %.3f,\n  \"wall_p999_us\": %.3f,\n  \"wall_max_us\": %.3f,\n"
                   "  \"cpu_mean_us\": %.3f,\n  \"cpu_p50_us\": %.3f,\n  \"cpu_p90_us\": %.3f,\n"
                   "  \"cpu_p99_us\": %.3f,\n  \"cpu_p999_us\": %.3f,\n  \"cpu_max_us\": %.3f,\n"
                   "  \"qps_1thread_wall\": %.1f,\n  \"qps_1thread_cpu\": %.1f,\n  \"threads\": %u,\n"
                   "  \"qps_multithread_wall\": %.1f,\n  \"mean_matches\": %.3f,\n"
                   "  \"postings_per_query\": %.1f,\n  \"cpu_ns_per_posting\": %.4f\n}\n",
                   mode.c_str(), cli::json_escape(a.get("index")).c_str(), ix.num_docs(), texts.size(),
                   reps, k, load0, mean, p50, p90, p99, p999, pmax, cmean, c50, c90, c99, c999, cmax,
                   1e6 / mean, 1e6 / cmean, threads, qps_mt,
                   static_cast<double>(total_matches) / static_cast<double>(texts.size()),
                   postings_per_query, cmean * 1e3 / std::max(1.0, postings_per_query));
      std::fclose(f);
    }
    if (a.has("csv")) {
      FILE* f = std::fopen(a.get("csv").c_str(), "w");
      if (!f) cli::die("cannot write csv");
      std::fprintf(f, "query_index,wall_us,cpu_us\n");
      for (size_t i = 0; i < first_wall.size(); ++i)
        std::fprintf(f, "%zu,%.3f,%.3f\n", i, first_wall[i], first_cpu[i]);
      std::fclose(f);
    }
  } catch (const std::exception& e) {
    cli::die(e.what());
  }
  return 0;
}
