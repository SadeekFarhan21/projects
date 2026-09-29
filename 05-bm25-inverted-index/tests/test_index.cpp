#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <random>

#include "se/format.hpp"
#include "test_util.hpp"

using namespace se;
using setest::build;
using setest::Doc;
using setest::plain;

TEST(Index, BuildsAndReadsBackSmallCorpus) {
  auto b = build({{"d0", "Red fox", "the quick red fox jumps"},
                  {"d1", "", "lazy dog sleeps"},
                  {"d2", "Fox news", "fox fox fox"}},
                 plain());
  const Index& ix = b->index;
  EXPECT_EQ(ix.num_docs(), 3u);
  EXPECT_EQ(ix.doc_id(0), "d0");
  EXPECT_EQ(ix.doc_id(2), "d2");
  EXPECT_EQ(ix.doc_len(0), 7u);  // 2 title + 5 text tokens
  EXPECT_EQ(ix.doc_len(1), 3u);
  EXPECT_DOUBLE_EQ(ix.avg_doc_len(), (7.0 + 3.0 + 5.0) / 3.0);

  const TermEntry* fox = ix.find("fox");
  ASSERT_NE(fox, nullptr);
  EXPECT_EQ(fox->df, 2u);
  EXPECT_EQ(fox->cf, 6u);
  auto c = ix.cursor(*fox);
  std::vector<uint32_t> pos;
  ASSERT_EQ(c.doc(), 0u);
  EXPECT_EQ(c.tf(), 2u);
  c.positions(pos);
  // title "red fox" -> 0,1 ; gap at 2 ; text starts at 3: the(3) quick(4) red(5) fox(6)
  EXPECT_EQ(pos, (std::vector<uint32_t>{1, 6}));
  c.next();
  ASSERT_EQ(c.doc(), 2u);
  EXPECT_EQ(c.tf(), 4u);
  c.positions(pos);
  EXPECT_EQ(pos, (std::vector<uint32_t>{0, 3, 4, 5}));
  c.positions(pos);  // asking twice is allowed
  EXPECT_EQ(pos, (std::vector<uint32_t>{0, 3, 4, 5}));
  c.next();
  EXPECT_TRUE(c.at_end());

  EXPECT_EQ(ix.find("cat"), nullptr);
  EXPECT_EQ(ix.find(""), nullptr);
  EXPECT_EQ(ix.find("zzzz"), nullptr);
  EXPECT_EQ(ix.tokenizer_options().stem, false);
}

TEST(Index, LexiconIsSortedAndComplete) {
  auto b = build({{"a", "", "delta alpha charlie"}, {"b", "", "bravo alpha echo"}}, plain());
  std::vector<std::string> terms;
  for (uint32_t i = 0; i < b->index.num_terms(); ++i)
    terms.emplace_back(b->index.term(b->index.term_entry(i)));
  EXPECT_EQ(terms, (std::vector<std::string>{"alpha", "bravo", "charlie", "delta", "echo"}));
}

TEST(Index, EmptyCorpus) {
  auto b = build({}, plain());
  EXPECT_EQ(b->index.num_docs(), 0u);
  EXPECT_EQ(b->index.num_terms(), 0u);
  EXPECT_EQ(b->index.find("x"), nullptr);
}

// Property test: random corpus, cursors must reproduce a brute-force
// inversion exactly, including lazily skipped positions and large doc gaps.
TEST(Index, RandomCorpusMatchesBruteForce) {
  std::mt19937 rng(7);
  std::vector<Doc> docs;
  const int ndocs = 3000;
  for (int d = 0; d < ndocs; ++d) {
    std::string text;
    int len = static_cast<int>(rng() % 40);
    for (int i = 0; i < len; ++i) {
      // Zipf-ish: low ids common, high ids rare, so gaps span 1..3 vbyte bytes
      uint32_t w = static_cast<uint32_t>(std::pow(rng() % 1000 / 1000.0, 3) * 300);
      text += "w" + std::to_string(w) + " ";
    }
    if (d == 0 || d == ndocs - 1) text += "bookend";
    docs.push_back({"doc" + std::to_string(d), "", text});
  }
  auto b = build(docs, plain());
  const Index& ix = b->index;

  // brute force: term -> doc -> positions
  std::map<std::string, std::map<uint32_t, std::vector<uint32_t>>> truth;
  Tokenizer tok(plain());
  for (uint32_t d = 0; d < docs.size(); ++d) {
    std::vector<Token> t;
    tok.tokenize(docs[d].text, t);
    for (auto& x : t) truth[x.term][d].push_back(x.pos);
  }
  ASSERT_EQ(ix.num_terms(), truth.size());
  std::vector<uint32_t> pos;
  for (auto& [term, postings] : truth) {
    const TermEntry* e = ix.find(term);
    ASSERT_NE(e, nullptr) << term;
    ASSERT_EQ(e->df, postings.size());
    auto c = ix.cursor(*e);
    size_t i = 0;
    for (auto& [d, p] : postings) {
      ASSERT_EQ(c.doc(), d);
      ASSERT_EQ(c.tf(), p.size());
      if ((i++ % 3) == 0) {  // read positions for only some docs
        c.positions(pos);
        ASSERT_EQ(pos, p);
      }
      c.next();
    }
    EXPECT_TRUE(c.at_end());
  }
  const TermEntry* be = ix.find("bookend");
  auto c = ix.cursor(*be);
  c.seek(1);
  EXPECT_EQ(c.doc(), static_cast<uint32_t>(ndocs - 1));
  c.seek(kEndDoc);
  EXPECT_TRUE(c.at_end());
}

static std::vector<char> slurp(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}
static void spit(const std::string& p, const std::vector<char>& d) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(d.data(), static_cast<std::streamsize>(d.size()));
}

TEST(Index, RejectsCorruptFiles) {
  auto b = build({{"a", "", "one two three"}, {"b", "", "two three four"}}, plain());
  auto bytes = slurp(b->tmp->path);
  setest::TempPath bad;

  EXPECT_THROW(Index::open(bad.path), std::runtime_error);  // missing file

  auto trunc = bytes;
  trunc.resize(trunc.size() - 8);
  spit(bad.path, trunc);
  EXPECT_THROW(Index::open(bad.path), std::runtime_error);

  auto magic = bytes;
  magic[0] = 'X';
  spit(bad.path, magic);
  EXPECT_THROW(Index::open(bad.path), std::runtime_error);

  // Corrupt the first byte of the postings section (doc stream of "four").
  Header h;
  std::memcpy(&h, bytes.data(), sizeof h);
  auto post = bytes;
  post[h.section_off[S_POSTINGS]] = static_cast<char>(0x80);  // turns a 1-byte gap into a truncated varint
  spit(bad.path, post);
  EXPECT_THROW(Index::open(bad.path), std::runtime_error);

  // Out-of-range doc id.
  auto range = bytes;
  range[h.section_off[S_POSTINGS]] = 0x05;
  spit(bad.path, range);
  EXPECT_THROW(Index::open(bad.path), std::runtime_error);
}
