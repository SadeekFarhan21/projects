#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "se/searcher.hpp"
#include "test_util.hpp"

using namespace se;
using setest::build;
using setest::Doc;
using setest::plain;

TEST(TopK, KeepsBestWithDeterministicTies) {
  TopK t(3);
  t.push(5, 1.0f);
  t.push(1, 2.0f);
  t.push(9, 1.0f);
  t.push(3, 1.0f);  // ties with 5 and 9 at 1.0; lower doc id wins
  t.push(7, 0.5f);
  auto h = t.take_sorted();
  ASSERT_EQ(h.size(), 3u);
  EXPECT_EQ(h[0].doc, 1u);
  EXPECT_EQ(h[1].doc, 3u);
  EXPECT_EQ(h[2].doc, 5u);
  TopK z(0);
  z.push(1, 1.0f);
  EXPECT_TRUE(z.take_sorted().empty());
}

TEST(Bm25, MatchesHandComputedFormula) {
  // N = 3, "fox" has df = 2. doc lens 7, 3, 5, avgdl = 5.
  auto b = build({{"d0", "Red fox", "the quick red fox jumps"},
                  {"d1", "", "lazy dog sleeps"},
                  {"d2", "Fox news", "fox fox fox"}},
                 plain());
  Searcher s(b->index, {.k1 = 1.2f, .b = 0.75f});
  const double N = 3, df = 2, avgdl = 5.0, k1 = 1.2, bb = 0.75;
  const double idf = std::log(1 + (N - df + 0.5) / (df + 0.5));
  auto bm25 = [&](double tf, double dl) {
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - bb + bb * dl / avgdl));
  };
  auto hits = s.search_bm25("fox", 10);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].doc, 2u);  // tf 4 in a shorter doc
  EXPECT_NEAR(hits[0].score, bm25(4, 5), 1e-5);
  EXPECT_EQ(hits[1].doc, 0u);
  EXPECT_NEAR(hits[1].score, bm25(2, 7), 1e-5);
  EXPECT_TRUE(s.search_bm25("unicorn", 10).empty());
  EXPECT_TRUE(s.search_bm25("", 10).empty());
}

TEST(Bm25, RepeatedQueryTermCountsTwice) {
  auto b = build({{"a", "", "x y"}, {"b", "", "y z"}, {"c", "", "q"}}, plain());
  Searcher s(b->index);
  auto one = s.search_bm25("x", 10), two = s.search_bm25("x x", 10);
  ASSERT_EQ(one.size(), 1u);
  EXPECT_NEAR(two[0].score, 2 * one[0].score, 1e-5);
}

// ------------------------------------------------------------------
// Randomised equivalence tests against a brute-force oracle built from the
// raw token lists (not from the index).

namespace {

struct Oracle {
  std::vector<std::vector<std::string>> toks;  // per doc, positions = index
  std::map<std::string, std::set<uint32_t>> docs_with;

  explicit Oracle(const std::vector<Doc>& docs) {
    Tokenizer t(plain());
    for (uint32_t d = 0; d < docs.size(); ++d) {
      toks.push_back(t.terms(docs[d].text));
      for (auto& w : toks.back()) docs_with[w].insert(d);
    }
  }
  std::set<uint32_t> term(const std::string& w) const {
    auto it = docs_with.find(w);
    return it == docs_with.end() ? std::set<uint32_t>{} : it->second;
  }
  std::set<uint32_t> phrase(const std::vector<std::string>& p) const {
    std::set<uint32_t> out;
    for (uint32_t d = 0; d < toks.size(); ++d) {
      const auto& t = toks[d];
      for (size_t i = 0; i + p.size() <= t.size(); ++i)
        if (std::equal(p.begin(), p.end(), t.begin() + static_cast<long>(i))) {
          out.insert(d);
          break;
        }
    }
    return out;
  }
  // Exact expected BM25 ranking, using the Searcher's own float formula so
  // results must match bit for bit (the formula itself is tested above).
  std::vector<Hit> bm25(const Searcher& s, const std::vector<std::string>& q, size_t k) const {
    std::vector<std::string> uniq;
    std::vector<float> w;
    for (auto& t : q) {
      auto df = static_cast<uint32_t>(term(t).size());
      float wt = df ? s.idf(df) * (s.params().k1 + 1.0f) : 0.0f;
      auto it = std::find(uniq.begin(), uniq.end(), t);
      if (it == uniq.end()) { uniq.push_back(t); w.push_back(wt); }
      else w[static_cast<size_t>(it - uniq.begin())] += wt;
    }
    std::vector<Hit> all;
    for (uint32_t d = 0; d < toks.size(); ++d) {
      float sc = 0.0f;
      bool any = false;
      for (size_t i = 0; i < uniq.size(); ++i) {
        auto tf = static_cast<uint32_t>(std::count(toks[d].begin(), toks[d].end(), uniq[i]));
        if (tf) { sc += s.term_score(w[i], tf, d); any = true; }
      }
      if (any) all.push_back({d, sc});
    }
    std::sort(all.begin(), all.end(), hit_better);
    if (all.size() > k) all.resize(k);
    return all;
  }
};

std::vector<Doc> random_docs(std::mt19937& rng, int n, int vocab) {
  std::vector<Doc> docs;
  for (int d = 0; d < n; ++d) {
    std::string text;
    int len = 1 + static_cast<int>(rng() % 30);
    for (int i = 0; i < len; ++i) {
      uint32_t w = static_cast<uint32_t>(std::pow((rng() % 1000) / 1000.0, 2) * vocab);
      text += "t" + std::to_string(w) + " ";
    }
    docs.push_back({std::to_string(d), "", text});
  }
  return docs;
}

void expect_same(const std::vector<Hit>& a, const std::vector<Hit>& b, const std::string& ctx) {
  ASSERT_EQ(a.size(), b.size()) << ctx;
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].doc, b[i].doc) << ctx << " rank " << i;
    EXPECT_EQ(a[i].score, b[i].score) << ctx << " rank " << i;
  }
}

}  // namespace

TEST(Bm25, DaatTaatAndOracleAgreeOnRandomCorpora) {
  std::mt19937 rng(123);
  for (int round = 0; round < 5; ++round) {
    auto docs = random_docs(rng, 400 + round * 300, 60);
    auto b = build(docs, plain());
    Searcher s(b->index);
    Oracle o(docs);
    for (int qi = 0; qi < 60; ++qi) {
      std::vector<std::string> q;
      std::string qs;
      int len = 1 + static_cast<int>(rng() % 5);
      for (int i = 0; i < len; ++i) {
        q.push_back("t" + std::to_string(rng() % 70));  // some terms are absent
        qs += q.back() + " ";
      }
      size_t k = 1 + rng() % 20;
      auto want = o.bm25(s, q, k);
      expect_same(s.search_bm25(qs, k), want, "daat " + qs);
      expect_same(s.search_bm25_taat(qs, k), want, "taat " + qs);
    }
  }
}

namespace {
// Random boolean tree rendered as query text, evaluated by the oracle.
std::string gen(std::mt19937& rng, const Oracle& o, int depth, std::set<uint32_t>& out) {
  int kind = depth <= 0 ? static_cast<int>(rng() % 2) : static_cast<int>(rng() % 4);
  auto word = [&] { return "t" + std::to_string(rng() % 12); };
  if (kind == 0) {
    std::string w = word();
    out = o.term(w);
    return w;
  }
  if (kind == 1) {
    std::vector<std::string> p;
    size_t len = 2 + rng() % 2;
    for (size_t i = 0; i < len; ++i) p.push_back(word());
    out = o.phrase(p);
    std::string s = "\"";
    for (size_t i = 0; i < p.size(); ++i) s += (i ? " " : "") + p[i];
    return s + "\"";
  }
  std::set<uint32_t> l, r;
  std::string ls = gen(rng, o, depth - 1, l), rs = gen(rng, o, depth - 1, r);
  out.clear();
  if (kind == 2) {
    std::set_intersection(l.begin(), l.end(), r.begin(), r.end(), std::inserter(out, out.end()));
    return "(" + ls + (rng() % 2 ? " AND " : " ") + rs + ")";
  }
  std::set_union(l.begin(), l.end(), r.begin(), r.end(), std::inserter(out, out.end()));
  return "(" + ls + " OR " + rs + ")";
}
}  // namespace

TEST(Boolean, RandomTreesMatchOracle) {
  std::mt19937 rng(99);
  // Small vocabulary so phrases actually occur.
  auto docs = random_docs(rng, 800, 12);
  auto b = build(docs, plain());
  Searcher s(b->index);
  Oracle o(docs);
  int nonempty = 0;
  for (int i = 0; i < 400; ++i) {
    std::set<uint32_t> want;
    std::string q = gen(rng, o, 3, want);
    auto tree = parse_boolean_query(q, s.tokenizer());
    ASSERT_TRUE(tree) << q;
    auto got = s.match(*tree);
    ASSERT_EQ(std::vector<uint32_t>(want.begin(), want.end()), got) << q;
    size_t total = 0;
    auto hits = s.search_boolean(q, 5, &total);
    EXPECT_EQ(total, want.size()) << q;
    EXPECT_EQ(hits.size(), std::min<size_t>(5, want.size())) << q;
    for (auto& h : hits) EXPECT_TRUE(want.count(h.doc)) << q;
    nonempty += !want.empty();
  }
  EXPECT_GT(nonempty, 100);  // the test is only meaningful if many queries match
}

TEST(Phrase, StopwordGapsAndFieldBoundary) {
  auto b = build({{"a", "", "lung cancer of the lung"},
                  {"b", "", "cancer lung"},
                  {"c", "cancer", "lung"},  // title/text boundary must not join
                  {"d", "", "cancer in the lung"}},
                 {.stem = false, .stopwords = true});
  Searcher s(b->index);
  auto ids = [&](const std::string& q) {
    std::vector<std::string> out;
    for (auto& h : s.search_boolean(q, 10)) out.emplace_back(b->index.doc_id(h.doc));
    std::sort(out.begin(), out.end());
    return out;
  };
  EXPECT_EQ(ids("\"cancer lung\""), (std::vector<std::string>{"b"}));
  // "of the" and "in the" are both two stopword slots, so both match.
  EXPECT_EQ(ids("\"cancer of the lung\""), (std::vector<std::string>{"a", "d"}));
  EXPECT_EQ(ids("\"lung cancer\""), (std::vector<std::string>{"a"}));
  EXPECT_EQ(ids("cancer AND lung"), (std::vector<std::string>{"a", "b", "c", "d"}));
  EXPECT_EQ(ids("\"cancer unicorn\""), (std::vector<std::string>{}));
}

TEST(Boolean, RankingUsesBm25OverQueryTerms) {
  auto b = build({{"a", "", "fox"}, {"b", "", "fox dog"}, {"c", "", "dog"}}, plain());
  Searcher s(b->index);
  size_t total = 0;
  auto hits = s.search_boolean("fox OR dog", 10, &total);
  EXPECT_EQ(total, 3u);
  auto ranked = s.search_bm25("fox dog", 10);
  expect_same(hits, ranked, "boolean OR == disjunctive bm25");
}
