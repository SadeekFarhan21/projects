#include <gtest/gtest.h>

#include "se/query.hpp"

using namespace se;

static std::string P(const std::string& q, TokenizerOptions o = {.stem = false, .stopwords = true}) {
  Tokenizer t(o);
  auto n = parse_boolean_query(q, t);
  return n ? to_string(*n) : "<empty>";
}

// Note: "a" is itself a stopword, so structural tests turn stopwords off.
static std::string S(const std::string& q) { return P(q, {.stem = false, .stopwords = false}); }

TEST(QueryParser, Precedence) {
  EXPECT_EQ(S("a b"), "(AND a b)");
  EXPECT_EQ(S("a AND b OR c"), "(OR (AND a b) c)");
  EXPECT_EQ(S("a OR b c"), "(OR a (AND b c))");
  EXPECT_EQ(S("(a OR b) c"), "(AND (OR a b) c)");
  EXPECT_EQ(S("a AND (b AND c)"), "(AND a b c)");  // flattened
  EXPECT_EQ(S("((a))"), "a");
}

TEST(QueryParser, PhrasesKeepStopwordGaps) {
  EXPECT_EQ(P("\"cancer of the lung\""), "\"cancer _ _ lung\"");
  EXPECT_EQ(P("\"new york\" OR nyc"), "(OR \"new york\" nyc)");
  EXPECT_EQ(P("covid-19"), "\"covid 19\"");  // multi-term word becomes a phrase
  EXPECT_EQ(P("\"the\""), "<empty>");
}

TEST(QueryParser, LowercaseOperatorsAreTerms) {
  EXPECT_EQ(P("rock or roll", {.stem = false, .stopwords = false}), "(AND rock or roll)");
  EXPECT_EQ(P("rock or roll"), "(AND rock roll)");  // "or" is a stopword
}

TEST(QueryParser, DropsStopwordOnlyOperands) {
  EXPECT_EQ(P("the AND fox"), "fox");
  EXPECT_EQ(P("the OR a"), "<empty>");
  EXPECT_EQ(P(""), "<empty>");
}

TEST(QueryParser, SyntaxErrors) {
  Tokenizer t;
  for (const char* bad : {"a AND", "OR a", "a OR", "(a b", "a b)", "\"a b", "()", "a AND OR b"})
    EXPECT_THROW(parse_boolean_query(bad, t), QueryParseError) << bad;
}

TEST(QueryParser, CollectTerms) {
  Tokenizer t({.stem = false, .stopwords = false});
  auto n = parse_boolean_query("(a OR \"b a c\") AND d", t);
  std::vector<std::string> terms;
  collect_terms(*n, terms);
  EXPECT_EQ(terms, (std::vector<std::string>{"a", "b", "c", "d"}));
}
