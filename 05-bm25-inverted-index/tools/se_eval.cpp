// Evaluate BM25 ranking quality against BEIR relevance judgments.
//
//   se_eval --index scifact.idx --queries queries.jsonl --qrels qrels/test.tsv
//           [--k1 0.9 --b 0.4] [--depth 100] [--ignore-identical-ids]
//           [--run out.trec] [--json out.json] [--per-query out.csv]
//
// Only queries that have judgments are evaluated (BEIR convention).
// --ignore-identical-ids drops a hit whose doc id equals the query id, which
// BEIR does for Quora (queries also appear in the corpus).
#include <cstdio>
#include <unordered_map>

#include "cli.hpp"
#include "se/corpus.hpp"
#include "se/metrics.hpp"
#include "se/searcher.hpp"

int main(int argc, char** argv) {
  auto a = cli::parse(argc, argv, {"ignore-identical-ids", "quiet"});
  if (a.get("index").empty() || a.get("queries").empty() || a.get("qrels").empty())
    cli::die("usage: se_eval --index f.idx --queries q.jsonl --qrels qrels.tsv [options]");
  try {
    auto ix = se::Index::open(a.get("index"));
    se::Bm25Params bp;
    bp.k1 = static_cast<float>(a.num("k1", bp.k1));
    bp.b = static_cast<float>(a.num("b", bp.b));
    se::Searcher s(ix, bp);
    auto queries = se::load_queries_jsonl(a.get("queries"));
    auto qrels = se::load_qrels_tsv(a.get("qrels"));
    const size_t depth = static_cast<size_t>(a.num("depth", 100));
    const bool ignore_ids = a.has("ignore-identical-ids");

    FILE* run = a.has("run") ? std::fopen(a.get("run").c_str(), "w") : nullptr;
    FILE* pq = a.has("per-query") ? std::fopen(a.get("per-query").c_str(), "w") : nullptr;
    if (pq) std::fprintf(pq, "query_id,ndcg@10,mrr@10,recall@100,num_relevant\n");

    double sn = 0, sm = 0, sr = 0;
    size_t nq = 0;
    double t0 = cli::now_s();
    for (const auto& q : queries) {
      auto it = qrels.find(q.id);
      if (it == qrels.end()) continue;
      auto hits = s.search_bm25(q.text, depth + (ignore_ids ? 1 : 0));
      std::vector<std::string> ranked;
      for (const auto& h : hits) {
        std::string id(ix.doc_id(h.doc));
        if (ignore_ids && id == q.id) continue;
        if (ranked.size() == depth) break;
        if (run)
          std::fprintf(run, "%s Q0 %s %zu %.6f se\n", q.id.c_str(), id.c_str(), ranked.size() + 1,
                       static_cast<double>(h.score));
        ranked.push_back(std::move(id));
      }
      double n10 = se::ndcg_at_k(ranked, it->second, 10);
      double m10 = se::mrr_at_k(ranked, it->second, 10);
      double r100 = se::recall_at_k(ranked, it->second, 100);
      if (pq) std::fprintf(pq, "%s,%.6f,%.6f,%.6f,%zu\n", q.id.c_str(), n10, m10, r100, it->second.size());
      sn += n10;
      sm += m10;
      sr += r100;
      ++nq;
    }
    double secs = cli::now_s() - t0;
    if (run) std::fclose(run);
    if (pq) std::fclose(pq);
    if (nq == 0) cli::die("no judged queries found");
    const double ndcg = sn / nq, mrr = sm / nq, rec = sr / nq;
    if (!a.has("quiet"))
      std::printf("queries=%zu k1=%.2f b=%.2f nDCG@10=%.4f MRR@10=%.4f Recall@100=%.4f (%.2fs)\n", nq,
                  static_cast<double>(bp.k1), static_cast<double>(bp.b), ndcg, mrr, rec, secs);
    else
      std::printf("%.2f,%.2f,%.6f,%.6f,%.6f\n", static_cast<double>(bp.k1),
                  static_cast<double>(bp.b), ndcg, mrr, rec);
    if (a.has("json")) {
      FILE* f = std::fopen(a.get("json").c_str(), "w");
      if (!f) cli::die("cannot write json");
      std::fprintf(f,
                   "{\n  \"index\": \"%s\",\n  \"qrels\": \"%s\",\n  \"k1\": %.3f,\n  \"b\": %.3f,\n"
                   "  \"num_queries\": %zu,\n  \"ndcg@10\": %.6f,\n  \"mrr@10\": %.6f,\n"
                   "  \"recall@100\": %.6f,\n  \"seconds\": %.4f\n}\n",
                   cli::json_escape(a.get("index")).c_str(), cli::json_escape(a.get("qrels")).c_str(),
                   static_cast<double>(bp.k1), static_cast<double>(bp.b), nq, ndcg, mrr, rec, secs);
      std::fclose(f);
    }
  } catch (const std::exception& e) {
    cli::die(e.what());
  }
  return 0;
}
