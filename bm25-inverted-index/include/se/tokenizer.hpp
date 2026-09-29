#pragma once
// Tokenizer and normaliser shared by indexing and querying.
//
// Rules (the same for documents and queries, which is the whole point):
//  * A token is a maximal run of ASCII letters, ASCII digits, or bytes >= 0x80
//    (so UTF-8 sequences stay inside tokens). Everything else separates.
//  * ASCII letters are lowercased. Non-ASCII bytes are kept verbatim.
//  * Tokens longer than kMaxTokenBytes are dropped.
//  * Optional stopword removal (Lucene's 33-word English list).
//  * Optional Porter stemming (only applied to pure a-z tokens).
//  * Every raw token consumes one position, including dropped stopwords, so a
//    phrase like "cancer of the lung" still requires the right gaps.
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace se {

struct TokenizerOptions {
  bool stem = true;
  bool stopwords = true;
};

struct Token {
  std::string term;
  uint32_t pos;
};

inline constexpr size_t kMaxTokenBytes = 64;

bool is_stopword(std::string_view term);

class Tokenizer {
 public:
  explicit Tokenizer(TokenizerOptions opts = {}) : opts_(opts) {}

  // Appends tokens of `text` to `out`, numbering positions from `start_pos`.
  // Returns the next unused position.
  uint32_t tokenize(std::string_view text, std::vector<Token>& out, uint32_t start_pos = 0) const;

  // Convenience: just the terms.
  std::vector<std::string> terms(std::string_view text) const;

  const TokenizerOptions& options() const { return opts_; }

 private:
  TokenizerOptions opts_;
};

}  // namespace se
