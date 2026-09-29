#pragma once
// Single-pass, in-memory inversion. Documents must be added in the order they
// should be numbered; postings are vbyte-encoded as they arrive, so memory use
// is roughly the size of the final compressed index plus the term dictionary.
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "se/tokenizer.hpp"

namespace se {

struct BuildStats {
  uint32_t num_docs = 0;
  uint32_t num_terms = 0;
  uint64_t num_postings = 0;
  uint64_t num_positions = 0;
  uint64_t file_bytes = 0;
  uint64_t section_bytes[6] = {};
  uint64_t doc_stream_bytes = 0;
  uint64_t pos_stream_bytes = 0;
  // What the same postings would take as plain uint32 arrays
  // (doc id + tf per posting, one uint32 per position).
  uint64_t raw_doc_stream_bytes = 0;
  uint64_t raw_pos_stream_bytes = 0;
};

class IndexBuilder {
 public:
  explicit IndexBuilder(TokenizerOptions opts = {}) : tok_(opts) {}

  // Tokenizes title and text (text positions start one past the title so a
  // phrase cannot straddle the field boundary) and adds the document.
  uint32_t add_document(std::string_view ext_id, std::string_view title, std::string_view text);

  // Tokenization is the expensive part of a build and is pure, so callers can
  // run it on worker threads and feed the results here in document order.
  static std::vector<Token> tokenize_document(const Tokenizer& tok, std::string_view title,
                                              std::string_view text);
  uint32_t add_tokenized(std::string_view ext_id, const std::vector<Token>& tokens);

  const Tokenizer& tokenizer() const { return tok_; }
  uint32_t num_docs() const { return static_cast<uint32_t>(doc_lens_.size()); }

  // Writes the index file. Throws std::runtime_error on I/O failure.
  BuildStats write(const std::string& path) const;

 private:
  struct TermPostings {
    std::vector<uint8_t> docs;
    std::vector<uint8_t> pos;
    uint32_t last_doc = 0;
    uint32_t df = 0;
    uint64_t cf = 0;
  };

  struct SvHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
  };

  Tokenizer tok_;
  std::unordered_map<std::string, uint32_t, SvHash, std::equal_to<>> term_ids_;
  std::vector<std::string> terms_;
  std::vector<TermPostings> postings_;
  std::vector<uint32_t> doc_lens_;
  std::vector<std::string> ext_ids_;
  uint64_t total_len_ = 0;
  // scratch reused across documents
  std::vector<std::pair<uint32_t, uint32_t>> scratch_;
};

}  // namespace se
