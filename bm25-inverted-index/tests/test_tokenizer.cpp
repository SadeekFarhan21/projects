#include <gtest/gtest.h>

#include "se/porter.hpp"
#include "se/tokenizer.hpp"

using namespace se;

static std::string stem(std::string w) {
  porter_stem(w);
  return w;
}

TEST(Porter, ReferenceVocabulary) {
  // Pairs from Porter's published voc.txt/output.txt test data.
  const std::pair<const char*, const char*> cases[] = {
      {"caresses", "caress"},   {"ponies", "poni"},         {"ties", "ti"},
      {"caress", "caress"},     {"cats", "cat"},            {"feed", "feed"},
      {"agreed", "agre"},       {"plastered", "plaster"},   {"bled", "bled"},
      {"motoring", "motor"},    {"sing", "sing"},           {"conflated", "conflat"},
      {"troubled", "troubl"},   {"sized", "size"},          {"hopping", "hop"},
      {"tanned", "tan"},        {"falling", "fall"},        {"hissing", "hiss"},
      {"fizzed", "fizz"},       {"failing", "fail"},        {"filing", "file"},
      {"happy", "happi"},       {"sky", "sky"},             {"relational", "relat"},
      {"conditional", "condit"},{"rational", "ration"},     {"digitizer", "digit"},
      {"generalizations", "gener"}, {"oscillators", "oscil"}, {"connection", "connect"},
      {"connections", "connect"},   {"connected", "connect"},  {"connecting", "connect"},
      {"running", "run"},       {"electricity", "electr"},  {"hopeful", "hope"},
      {"goodness", "good"},     {"adjustable", "adjust"},   {"effective", "effect"},
      {"probate", "probat"},    {"rate", "rate"},           {"cease", "ceas"},
      {"controlling", "control"}, {"roll", "roll"},         {"analogous", "analog"},
  };
  for (auto [in, want] : cases) EXPECT_EQ(stem(in), want) << in;
}

TEST(Porter, LeavesNonAlphaAndShortWordsAlone) {
  EXPECT_EQ(stem("is"), "is");
  EXPECT_EQ(stem("covid19"), "covid19");
  EXPECT_EQ(stem("caf\xC3\xA9s"), "caf\xC3\xA9s");
}

TEST(Tokenizer, SplitsLowercasesAndNumbersPositions) {
  Tokenizer t({.stem = false, .stopwords = false});
  std::vector<Token> out;
  uint32_t next = t.tokenize("Hello, WORLD!  covid-19 (x2)", out);
  ASSERT_EQ(out.size(), 5u);
  EXPECT_EQ(out[0].term, "hello");
  EXPECT_EQ(out[1].term, "world");
  EXPECT_EQ(out[2].term, "covid");
  EXPECT_EQ(out[3].term, "19");
  EXPECT_EQ(out[4].term, "x2");
  for (uint32_t i = 0; i < 5; ++i) EXPECT_EQ(out[i].pos, i);
  EXPECT_EQ(next, 5u);
}

TEST(Tokenizer, StopwordsKeepTheirPositionSlots) {
  Tokenizer t({.stem = false, .stopwords = true});
  std::vector<Token> out;
  t.tokenize("The cancer of the lung", out);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].term, "cancer");
  EXPECT_EQ(out[0].pos, 1u);
  EXPECT_EQ(out[1].term, "lung");
  EXPECT_EQ(out[1].pos, 4u);
}

TEST(Tokenizer, StemsWhenEnabled) {
  Tokenizer t({.stem = true, .stopwords = true});
  EXPECT_EQ(t.terms("Running connections are effective"),
            (std::vector<std::string>{"run", "connect", "effect"}));
}

TEST(Tokenizer, Utf8BytesStayInsideTokens) {
  Tokenizer t({.stem = true, .stopwords = false});
  EXPECT_EQ(t.terms("Caf\xC3\xA9 na\xC3\xAFve"),
            (std::vector<std::string>{"caf\xC3\xA9", "na\xC3\xAFve"}));
}

TEST(Tokenizer, DropsOverlongTokensButCountsThem) {
  Tokenizer t({.stem = false, .stopwords = false});
  std::vector<Token> out;
  t.tokenize("a " + std::string(100, 'x') + " b", out);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[1].term, "b");
  EXPECT_EQ(out[1].pos, 2u);
}

TEST(Tokenizer, EmptyAndSeparatorOnlyInput) {
  Tokenizer t;
  EXPECT_TRUE(t.terms("").empty());
  EXPECT_TRUE(t.terms(" ,.;!? ").empty());
}
