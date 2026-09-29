#pragma once
// Read-only view of an index file, memory-mapped. Opening is O(num_terms +
// total postings bytes) because every list is validated once; after that all
// reads are unchecked and zero-copy.
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "se/format.hpp"
#include "se/tokenizer.hpp"
#include "se/varint.hpp"

namespace se {

inline constexpr uint32_t kEndDoc = std::numeric_limits<uint32_t>::max();

// Forward iterator over one postings list (document-at-a-time).
// Positions live in a separate stream and are decoded lazily: moving past a
// document without reading its positions only records how many position
// values to skip, so BM25-only traversal never touches the position stream.
class PostingCursor {
 public:
  PostingCursor() = default;
  PostingCursor(const uint8_t* docs, const uint8_t* pos, uint32_t df)
      : p_(docs), pp_(pos), remaining_(df), doc_(0) {
    next();  // doc_ starts at 0, so the first gap decodes to the absolute id
  }

  uint32_t doc() const { return doc_; }
  uint32_t tf() const { return tf_; }
  bool at_end() const { return doc_ == kEndDoc; }

  void next() {
    if (!pos_read_) pending_skip_ += tf_;
    pos_read_ = false;
    if (remaining_ == 0) {
      doc_ = kEndDoc;
      tf_ = 0;
      return;
    }
    --remaining_;
    doc_ += vbyte_decode(p_);
    tf_ = vbyte_decode(p_);
  }

  // Advance to the first posting with doc >= target (no-op if already there).
  // Linear: v0 has no skip pointers (see ROADMAP in README).
  void seek(uint32_t target) {
    while (doc_ < target) next();
  }

  // Positions of the current document, ascending. Only valid when !at_end().
  // Safe to call more than once for the same document.
  void positions(std::vector<uint32_t>& out) {
    out.clear();
    if (pos_read_) {
      pp_ = cur_pos_start_;  // rewind and decode again
    } else {
      vbyte_skip(pp_, pending_skip_);
      pending_skip_ = 0;
      cur_pos_start_ = pp_;
    }
    uint32_t p = 0;
    for (uint32_t i = 0; i < tf_; ++i) {
      p += vbyte_decode(pp_);
      out.push_back(p);
    }
    pos_read_ = true;
  }

 private:
  const uint8_t* p_ = nullptr;
  const uint8_t* pp_ = nullptr;
  uint32_t remaining_ = 0;
  uint32_t doc_ = kEndDoc;
  uint32_t tf_ = 0;
  uint32_t pending_skip_ = 0;
  bool pos_read_ = false;
  const uint8_t* cur_pos_start_ = nullptr;
};

class Index {
 public:
  // Throws std::runtime_error if the file is missing, truncated or corrupt.
  static Index open(const std::string& path);

  Index(Index&& o) noexcept;
  Index& operator=(Index&& o) noexcept;
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  ~Index();

  uint32_t num_docs() const { return h_->num_docs; }
  uint32_t num_terms() const { return h_->num_terms; }
  uint64_t num_postings() const { return h_->num_postings; }
  uint64_t file_size() const { return size_; }
  double avg_doc_len() const {
    return h_->num_docs ? static_cast<double>(h_->total_doc_len) / h_->num_docs : 0.0;
  }
  uint32_t doc_len(uint32_t d) const { return doc_lens_[d]; }
  std::string_view doc_id(uint32_t d) const;
  TokenizerOptions tokenizer_options() const;
  const Header& header() const { return *h_; }

  // Binary search in the sorted lexicon. Returns nullptr when absent.
  const TermEntry* find(std::string_view term) const;
  const TermEntry& term_entry(uint32_t i) const { return entries_[i]; }
  std::string_view term(const TermEntry& e) const;

  PostingCursor cursor(const TermEntry& e) const {
    return PostingCursor(post_ + e.docs_off, post_ + e.pos_off, e.df);
  }

 private:
  Index() = default;
  void validate() const;

  void* map_ = nullptr;
  size_t size_ = 0;
  const uint8_t* base_ = nullptr;
  const Header* h_ = nullptr;
  const uint32_t* doc_lens_ = nullptr;
  const uint64_t* docid_offs_ = nullptr;
  const char* docid_blob_ = nullptr;
  const TermEntry* entries_ = nullptr;
  const char* strings_ = nullptr;
  const uint8_t* post_ = nullptr;
};

}  // namespace se
