#include "se/tokenizer.hpp"

#include <algorithm>
#include <array>

#include "se/porter.hpp"

namespace se {

namespace {
// Lucene EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, sorted for binary search.
constexpr std::array<std::string_view, 33> kStopwords = {
    "a",    "an",    "and",   "are",  "as",   "at",   "be",    "but",  "by",
    "for",  "if",    "in",    "into", "is",   "it",   "no",    "not",  "of",
    "on",   "or",    "such",  "that", "the",  "their", "then", "there", "these",
    "they", "this",  "to",    "was",  "will", "with"};

inline bool is_token_byte(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80;
}
}  // namespace

bool is_stopword(std::string_view term) {
  return std::binary_search(kStopwords.begin(), kStopwords.end(), term);
}

uint32_t Tokenizer::tokenize(std::string_view text, std::vector<Token>& out,
                             uint32_t pos) const {
  size_t i = 0, n = text.size();
  std::string buf;
  while (i < n) {
    while (i < n && !is_token_byte(static_cast<unsigned char>(text[i]))) ++i;
    if (i >= n) break;
    size_t start = i;
    while (i < n && is_token_byte(static_cast<unsigned char>(text[i]))) ++i;
    uint32_t this_pos = pos++;
    if (i - start > kMaxTokenBytes) continue;
    buf.assign(text.data() + start, i - start);
    for (char& c : buf)
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (opts_.stopwords && is_stopword(buf)) continue;
    if (opts_.stem) porter_stem(buf);
    out.push_back(Token{buf, this_pos});
  }
  return pos;
}

std::vector<std::string> Tokenizer::terms(std::string_view text) const {
  std::vector<Token> toks;
  tokenize(text, toks);
  std::vector<std::string> out;
  out.reserve(toks.size());
  for (auto& t : toks) out.push_back(std::move(t.term));
  return out;
}

}  // namespace se
