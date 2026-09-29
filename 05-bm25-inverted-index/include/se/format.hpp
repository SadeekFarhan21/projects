#pragma once
// On-disk index layout. One file, little-endian, every section 8-byte aligned.
//
//   [Header]
//   [S_DOC_LENS]     uint32 doc_len[num_docs]           indexed tokens per doc
//   [S_DOCID_OFFS]   uint64 off[num_docs + 1]           into S_DOCID_BLOB
//   [S_DOCID_BLOB]   external doc ids, concatenated
//   [S_LEX_ENTRIES]  TermEntry[num_terms], sorted by term bytes
//   [S_LEX_STRINGS]  term bytes, concatenated
//   [S_POSTINGS]     per term: doc stream, then position stream
//
// Doc stream (df entries):   vbyte(doc - prev_doc) vbyte(tf)     prev_doc starts at 0
// Position stream (df runs): for each posting, tf values:
//                            vbyte(first_pos) vbyte(pos_i - pos_{i-1}) ...
// Doc ids strictly increase inside a list, so every gap after the first is >= 1.
#include <cstdint>

namespace se {

inline constexpr char kMagic[8] = {'S', 'E', 'I', 'D', 'X', '0', '0', '1'};
inline constexpr uint32_t kFormatVersion = 1;

enum Section : int {
  S_DOC_LENS = 0,
  S_DOCID_OFFS,
  S_DOCID_BLOB,
  S_LEX_ENTRIES,
  S_LEX_STRINGS,
  S_POSTINGS,
  S_COUNT
};

enum HeaderFlags : uint32_t { F_STEM = 1u << 0, F_STOPWORDS = 1u << 1 };

struct Header {
  char magic[8];
  uint32_t version;
  uint32_t flags;
  uint32_t num_docs;
  uint32_t num_terms;
  uint64_t total_doc_len;  // sum of doc_len, for avgdl
  uint64_t num_postings;   // sum of df
  uint64_t num_positions;  // sum of tf
  uint64_t section_off[S_COUNT];
  uint64_t section_len[S_COUNT];
  uint64_t file_size;
};

struct TermEntry {
  uint64_t str_off;
  uint64_t docs_off;   // absolute offset of the doc stream within S_POSTINGS
  uint64_t pos_off;    // absolute offset of the position stream within S_POSTINGS
  uint64_t cf;         // collection frequency (sum of tf)
  uint32_t str_len;
  uint32_t df;
  uint32_t docs_len;   // bytes
  uint32_t pos_len;    // bytes
};

static_assert(sizeof(TermEntry) == 48);

}  // namespace se
