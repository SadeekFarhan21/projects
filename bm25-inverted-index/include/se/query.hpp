#pragma once
// Boolean query language.
//
//   expr    := and_expr ( "OR" and_expr )*
//   and_expr:= unary ( ["AND"] unary )*          juxtaposition means AND
//   unary   := "(" expr ")" | '"' words '"' | word
//
// Operators are case sensitive (AND, OR) so the words "and"/"or" inside a
// query are ordinary terms. Words go through the index's tokenizer: a word
// that normalises to several terms (e.g. "covid-19") becomes a phrase, and a
// word that normalises to nothing (a stopword) is dropped.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "se/tokenizer.hpp"

namespace se {

struct QueryParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct QueryNode {
  enum class Kind { Term, Phrase, And, Or };
  Kind kind;
  // Term: terms[0]. Phrase: terms with their relative offsets (offsets[0] == 0;
  // gaps come from dropped stopwords).
  std::vector<std::string> terms;
  std::vector<uint32_t> offsets;
  std::vector<std::unique_ptr<QueryNode>> children;  // And / Or
};

// Returns nullptr when the query has no searchable terms. Throws
// QueryParseError on syntax errors (unbalanced parentheses or quotes,
// dangling operators).
std::unique_ptr<QueryNode> parse_boolean_query(std::string_view q, const Tokenizer& tok);

// S-expression rendering for tests and --explain: (AND a (OR b c) "d _ e").
std::string to_string(const QueryNode& n);

// All distinct terms mentioned anywhere in the tree, in first-seen order.
void collect_terms(const QueryNode& n, std::vector<std::string>& out);

}  // namespace se
