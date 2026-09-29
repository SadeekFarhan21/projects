// Query an index from the command line.
//
//   se_search --index scifact.idx [--mode bm25|taat|boolean] [--k 10]
//             [--corpus corpus.jsonl] [--explain] "query text"
//
// With no query argument it reads one query per line from stdin.
// --corpus is only used to print titles next to results.
#include <cstdio>
#include <iostream>
#include <unordered_map>

#include "cli.hpp"
#include "se/corpus.hpp"
#include "se/searcher.hpp"

int main(int argc, char** argv) {
  auto a = cli::parse(argc, argv, {"explain"});
  if (a.get("index").empty())
    cli::die("usage: se_search --index f.idx [--mode bm25|taat|boolean] [--k 10] [--corpus c.jsonl] [query]");
  try {
    auto ix = se::Index::open(a.get("index"));
    se::Bm25Params bp;
    bp.k1 = static_cast<float>(a.num("k1", bp.k1));
    bp.b = static_cast<float>(a.num("b", bp.b));
    se::Searcher s(ix, bp);
    const std::string mode = a.get("mode", "bm25");
    const size_t k = static_cast<size_t>(a.num("k", 10));

    std::unordered_map<std::string, std::string> titles;
    if (a.has("corpus"))
      for (auto& d : se::load_corpus_jsonl(a.get("corpus"))) titles[d.id] = d.title.substr(0, 90);

    auto run = [&](const std::string& q) {
      double t0 = cli::now_s();
      std::vector<se::Hit> hits;
      size_t matches = 0;
      if (mode == "bm25") hits = s.search_bm25(q, k);
      else if (mode == "taat") hits = s.search_bm25_taat(q, k);
      else if (mode == "boolean") {
        if (a.has("explain")) {
          auto tree = se::parse_boolean_query(q, s.tokenizer());
          std::printf("parsed: %s\n", tree ? se::to_string(*tree).c_str() : "(empty)");
        }
        hits = s.search_boolean(q, k, &matches);
      } else cli::die("unknown mode " + mode);
      double ms = (cli::now_s() - t0) * 1e3;
      if (mode == "boolean") std::printf("%zu matching docs, ", matches);
      std::printf("%.3f ms\n", ms);
      for (size_t i = 0; i < hits.size(); ++i) {
        auto id = std::string(ix.doc_id(hits[i].doc));
        std::printf("%3zu  %-12s %8.4f  %s\n", i + 1, id.c_str(), hits[i].score,
                    titles.count(id) ? titles[id].c_str() : "");
      }
    };
    if (!a.positional.empty()) {
      std::string q;
      for (auto& p : a.positional) q += (q.empty() ? "" : " ") + p;
      run(q);
    } else {
      std::string line;
      while (std::getline(std::cin, line))
        if (!line.empty()) run(line);
    }
  } catch (const se::QueryParseError& e) {
    cli::die(std::string("query syntax: ") + e.what());
  } catch (const std::exception& e) {
    cli::die(e.what());
  }
  return 0;
}
