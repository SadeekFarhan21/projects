#pragma once
// Query evaluation over an Index: BM25 ranking (DAAT and TAAT), boolean
// AND/OR, positional phrase matching, and top-k selection with a heap.
//
// A Searcher owns per-query scratch space, so use one per thread. The Index
// it points to is immutable and can be shared freely.
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "se/index.hpp"
#include "se/query.hpp"

namespace se {

struct Bm25Params {
  // Anserini/Pyserini defaults, which the BEIR BM25 baselines are close to.
  float k1 = 0.9f;
  float b = 0.4f;
};

struct Hit {
  uint32_t doc;
  float score;
};

// Ranking order used everywhere: higher score first, then lower doc id.
// Making ties deterministic is what lets DAAT and TAAT be compared exactly.
inline bool hit_better(const Hit& a, const Hit& b) {
  return a.score > b.score || (a.score == b.score && a.doc < b.doc);
}

// Bounded min-heap that keeps the k best hits seen so far.
class TopK {
 public:
  explicit TopK(size_t k) : k_(k) { heap_.reserve(k + 1); }
  // Score a new candidate must beat to enter (only meaningful when full()).
  bool full() const { return heap_.size() >= k_; }
  void push(uint32_t doc, float score);
  // Sorted best-first; leaves the heap empty.
  std::vector<Hit> take_sorted();

 private:
  size_t k_;
  std::vector<Hit> heap_;
};

struct WeightedTerm {
  const TermEntry* entry;  // nullptr when the term is not in the index
  std::string term;
  float weight;  // idf * (k1 + 1) * query term frequency
};

class Searcher {
 public:
  explicit Searcher(const Index& ix, Bm25Params p = {});

  const Index& index() const { return ix_; }
  const Bm25Params& params() const { return p_; }
  const Tokenizer& tokenizer() const { return tok_; }

  float idf(uint32_t df) const;

  // Bag-of-words BM25 over the union of query terms (disjunctive).
  // Document-at-a-time: one cursor per term, advance the minimum doc.
  std::vector<Hit> search_bm25(std::string_view query, size_t k);
  // Term-at-a-time with a dense float accumulator, then heap selection.
  std::vector<Hit> search_bm25_taat(std::string_view query, size_t k);

  // Boolean filtering (see query.hpp for syntax), matches ranked by BM25 over
  // every term that appears in the query. total_matches receives the size of
  // the matching set before top-k truncation.
  std::vector<Hit> search_boolean(std::string_view query, size_t k, size_t* total_matches = nullptr);

  // Every document matching the boolean tree, ascending.
  std::vector<uint32_t> match(const QueryNode& n);

  // Weighted, de-duplicated query terms in first-seen order.
  std::vector<WeightedTerm> weigh_terms(const std::vector<std::string>& terms) const;

  // BM25 contribution of one posting (exposed for tests).
  float term_score(float weight, uint32_t tf, uint32_t doc) const {
    const float t = static_cast<float>(tf);
    return weight * t / (t + norm_[doc]);
  }

 private:
  std::vector<uint32_t> match_phrase(const QueryNode& n);
  std::vector<uint32_t> match_conjunction(const std::vector<const TermEntry*>& terms);
  std::vector<Hit> rank_docs(const std::vector<uint32_t>& docs,
                             const std::vector<WeightedTerm>& wts, size_t k);

  const Index& ix_;
  Bm25Params p_;
  Tokenizer tok_;
  std::vector<float> norm_;   // k1 * (1 - b + b * dl / avgdl), per doc
  std::vector<float> accum_;  // TAAT scratch, kept all-zero between queries
  std::vector<uint32_t> touched_;
};

}  // namespace se
