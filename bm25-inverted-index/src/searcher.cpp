#include "se/searcher.hpp"

#include <algorithm>
#include <cmath>

namespace se {

// ---------------------------------------------------------------- TopK

namespace {
// Heap comparator: "a is better than b" makes std::push_heap build a heap
// whose front is the worst element, i.e. a min-heap in ranking order.
struct WorseOnTop {
  bool operator()(const Hit& a, const Hit& b) const { return hit_better(a, b); }
};
}  // namespace

void TopK::push(uint32_t doc, float score) {
  if (k_ == 0) return;
  Hit h{doc, score};
  if (heap_.size() < k_) {
    heap_.push_back(h);
    std::push_heap(heap_.begin(), heap_.end(), WorseOnTop{});
  } else if (hit_better(h, heap_.front())) {
    std::pop_heap(heap_.begin(), heap_.end(), WorseOnTop{});
    heap_.back() = h;
    std::push_heap(heap_.begin(), heap_.end(), WorseOnTop{});
  }
}

std::vector<Hit> TopK::take_sorted() {
  std::vector<Hit> out = std::move(heap_);
  heap_.clear();
  std::sort(out.begin(), out.end(), hit_better);
  return out;
}

// ---------------------------------------------------------------- Searcher

Searcher::Searcher(const Index& ix, Bm25Params p)
    : ix_(ix), p_(p), tok_(ix.tokenizer_options()) {
  const uint32_t n = ix.num_docs();
  const double avgdl = ix.avg_doc_len() > 0 ? ix.avg_doc_len() : 1.0;
  norm_.resize(n);
  for (uint32_t d = 0; d < n; ++d)
    norm_[d] = static_cast<float>(p_.k1 * (1.0 - p_.b + p_.b * ix.doc_len(d) / avgdl));
  accum_.assign(n, 0.0f);
}

// Lucene's BM25 idf, which is always positive (unlike the classic
// Robertson-Sparck Jones form, which goes negative for df > N/2).
float Searcher::idf(uint32_t df) const {
  const double N = ix_.num_docs();
  return static_cast<float>(std::log(1.0 + (N - df + 0.5) / (df + 0.5)));
}

std::vector<WeightedTerm> Searcher::weigh_terms(const std::vector<std::string>& terms) const {
  std::vector<WeightedTerm> out;
  for (const auto& t : terms) {
    auto it = std::find_if(out.begin(), out.end(), [&](const WeightedTerm& w) { return w.term == t; });
    const TermEntry* e = ix_.find(t);
    const float w = e ? idf(e->df) * (p_.k1 + 1.0f) : 0.0f;
    if (it != out.end()) it->weight += w;  // repeated query term: weight scales with qtf
    else out.push_back(WeightedTerm{e, t, w});
  }
  return out;
}

std::vector<Hit> Searcher::search_bm25(std::string_view query, size_t k) {
  auto wts = weigh_terms(tok_.terms(query));
  std::vector<PostingCursor> cur;
  std::vector<float> w;
  for (const auto& t : wts) {
    if (!t.entry) continue;
    cur.push_back(ix_.cursor(*t.entry));
    w.push_back(t.weight);
  }
  TopK top(k);
  uint32_t d = kEndDoc;
  for (const auto& c : cur) d = std::min(d, c.doc());
  while (d != kEndDoc) {
    // One pass per candidate: score the cursors sitting on d, advance them,
    // and find the next candidate (the minimum doc) at the same time.
    // Summing in query-term order keeps the float result bit-identical to TAAT.
    float s = 0.0f;
    uint32_t next_d = kEndDoc;
    for (size_t i = 0; i < cur.size(); ++i) {
      if (cur[i].doc() == d) {
        s += term_score(w[i], cur[i].tf(), d);
        cur[i].next();
      }
      next_d = std::min(next_d, cur[i].doc());
    }
    top.push(d, s);
    d = next_d;
  }
  return top.take_sorted();
}

std::vector<Hit> Searcher::search_bm25_taat(std::string_view query, size_t k) {
  auto wts = weigh_terms(tok_.terms(query));
  touched_.clear();
  for (const auto& t : wts) {
    if (!t.entry) continue;
    for (PostingCursor c = ix_.cursor(*t.entry); !c.at_end(); c.next()) {
      const uint32_t d = c.doc();
      if (accum_[d] == 0.0f) touched_.push_back(d);
      accum_[d] += term_score(t.weight, c.tf(), d);
    }
  }
  TopK top(k);
  for (uint32_t d : touched_) {
    top.push(d, accum_[d]);
    accum_[d] = 0.0f;  // restore the all-zero invariant for the next query
  }
  return top.take_sorted();
}

// ---------------------------------------------------------------- boolean

namespace {
std::vector<uint32_t> intersect(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  std::vector<uint32_t> out;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}
std::vector<uint32_t> unite(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  std::vector<uint32_t> out;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}
}  // namespace

// Leapfrog intersection over cursors, rarest list first so it drives.
std::vector<uint32_t> Searcher::match_conjunction(const std::vector<const TermEntry*>& terms) {
  std::vector<uint32_t> out;
  for (auto* e : terms)
    if (!e) return out;
  if (terms.empty()) return out;
  std::vector<const TermEntry*> sorted = terms;
  std::sort(sorted.begin(), sorted.end(),
            [](const TermEntry* a, const TermEntry* b) { return a->df < b->df; });
  std::vector<PostingCursor> cur;
  for (auto* e : sorted) cur.push_back(ix_.cursor(*e));
  uint32_t target = cur[0].doc();
  while (target != kEndDoc) {
    bool agreed = true;
    for (size_t i = 1; i < cur.size(); ++i) {
      cur[i].seek(target);
      if (cur[i].doc() != target) {
        target = cur[i].doc();
        agreed = false;
        break;
      }
    }
    if (agreed) {
      out.push_back(target);
      cur[0].next();
      target = cur[0].doc();
    } else {
      cur[0].seek(target);
      target = cur[0].doc();
    }
  }
  return out;
}

std::vector<uint32_t> Searcher::match_phrase(const QueryNode& n) {
  std::vector<uint32_t> out;
  const size_t m = n.terms.size();
  std::vector<const TermEntry*> ents(m);
  for (size_t i = 0; i < m; ++i) {
    ents[i] = ix_.find(n.terms[i]);
    if (!ents[i]) return out;
  }
  // Candidates: documents containing every term (a term may repeat in the
  // phrase, so use one cursor per phrase slot).
  std::vector<PostingCursor> cur;
  for (auto* e : ents) cur.push_back(ix_.cursor(*e));
  std::vector<uint32_t> cand, next_cand, pos;
  auto candidates = match_conjunction(ents);
  for (uint32_t d : candidates) {
    // cand holds start positions p such that slot 0..i all line up.
    cur[0].seek(d);
    cur[0].positions(cand);
    for (size_t i = 1; i < m && !cand.empty(); ++i) {
      cur[i].seek(d);
      cur[i].positions(pos);
      const uint32_t off = n.offsets[i];
      next_cand.clear();
      // Merge: keep p where p + off occurs in pos. Both sides ascending.
      size_t a = 0, b = 0;
      while (a < cand.size() && b < pos.size()) {
        const uint64_t want = static_cast<uint64_t>(cand[a]) + off;
        if (pos[b] < want) ++b;
        else if (pos[b] > want) ++a;
        else { next_cand.push_back(cand[a]); ++a; ++b; }
      }
      cand.swap(next_cand);
    }
    if (!cand.empty()) out.push_back(d);
  }
  return out;
}

std::vector<uint32_t> Searcher::match(const QueryNode& n) {
  switch (n.kind) {
    case QueryNode::Kind::Term: {
      std::vector<uint32_t> out;
      if (const TermEntry* e = ix_.find(n.terms[0])) {
        out.reserve(e->df);
        for (PostingCursor c = ix_.cursor(*e); !c.at_end(); c.next()) out.push_back(c.doc());
      }
      return out;
    }
    case QueryNode::Kind::Phrase:
      return match_phrase(n);
    case QueryNode::Kind::And: {
      // Fast path: a conjunction of plain terms streams through cursors
      // without materialising any list.
      bool all_terms = std::all_of(n.children.begin(), n.children.end(), [](const auto& c) {
        return c->kind == QueryNode::Kind::Term;
      });
      if (all_terms) {
        std::vector<const TermEntry*> ents;
        for (const auto& c : n.children) ents.push_back(ix_.find(c->terms[0]));
        return match_conjunction(ents);
      }
      std::vector<std::vector<uint32_t>> parts;
      for (const auto& c : n.children) parts.push_back(match(*c));
      std::sort(parts.begin(), parts.end(),
                [](const auto& a, const auto& b) { return a.size() < b.size(); });
      std::vector<uint32_t> acc = std::move(parts[0]);
      for (size_t i = 1; i < parts.size() && !acc.empty(); ++i) acc = intersect(acc, parts[i]);
      return acc;
    }
    case QueryNode::Kind::Or: {
      std::vector<uint32_t> acc;
      for (const auto& c : n.children) acc = unite(acc, match(*c));
      return acc;
    }
  }
  return {};
}

std::vector<Hit> Searcher::rank_docs(const std::vector<uint32_t>& docs,
                                     const std::vector<WeightedTerm>& wts, size_t k) {
  std::vector<PostingCursor> cur;
  std::vector<float> w;
  for (const auto& t : wts) {
    if (!t.entry) continue;
    cur.push_back(ix_.cursor(*t.entry));
    w.push_back(t.weight);
  }
  TopK top(k);
  for (uint32_t d : docs) {  // ascending, so every seek moves forward
    float s = 0.0f;
    for (size_t i = 0; i < cur.size(); ++i) {
      cur[i].seek(d);
      if (cur[i].doc() == d) s += term_score(w[i], cur[i].tf(), d);
    }
    top.push(d, s);
  }
  return top.take_sorted();
}

std::vector<Hit> Searcher::search_boolean(std::string_view query, size_t k, size_t* total_matches) {
  auto tree = parse_boolean_query(query, tok_);
  if (!tree) {
    if (total_matches) *total_matches = 0;
    return {};
  }
  auto docs = match(*tree);
  if (total_matches) *total_matches = docs.size();
  std::vector<std::string> terms;
  collect_terms(*tree, terms);
  return rank_docs(docs, weigh_terms(terms), k);
}

}  // namespace se
