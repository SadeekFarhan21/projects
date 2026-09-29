#include "se/index.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace se {

namespace {
[[noreturn]] void corrupt(const std::string& what) {
  throw std::runtime_error("corrupt index: " + what);
}
}  // namespace

Index Index::open(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open index " + path);
  struct stat sb {};
  if (::fstat(fd, &sb) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  Index ix;
  ix.size_ = static_cast<size_t>(sb.st_size);
  if (ix.size_ < sizeof(Header)) {
    ::close(fd);
    corrupt("file smaller than header");
  }
  void* m = ::mmap(nullptr, ix.size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
  ix.map_ = m;
  ix.base_ = static_cast<const uint8_t*>(m);
  ix.h_ = reinterpret_cast<const Header*>(ix.base_);
  ix.validate();  // throws; the destructor unmaps
  const Header& h = *ix.h_;
  ix.doc_lens_ = reinterpret_cast<const uint32_t*>(ix.base_ + h.section_off[S_DOC_LENS]);
  ix.docid_offs_ = reinterpret_cast<const uint64_t*>(ix.base_ + h.section_off[S_DOCID_OFFS]);
  ix.docid_blob_ = reinterpret_cast<const char*>(ix.base_ + h.section_off[S_DOCID_BLOB]);
  ix.entries_ = reinterpret_cast<const TermEntry*>(ix.base_ + h.section_off[S_LEX_ENTRIES]);
  ix.strings_ = reinterpret_cast<const char*>(ix.base_ + h.section_off[S_LEX_STRINGS]);
  ix.post_ = ix.base_ + h.section_off[S_POSTINGS];
  return ix;
}

void Index::validate() const {
  const Header& h = *h_;
  if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0) corrupt("bad magic");
  if (h.version != kFormatVersion) corrupt("unsupported version");
  if (h.file_size != size_) corrupt("size mismatch");
  for (int s = 0; s < S_COUNT; ++s) {
    if (h.section_off[s] % 8 != 0) corrupt("misaligned section");
    if (h.section_off[s] > size_ || h.section_len[s] > size_ - h.section_off[s])
      corrupt("section out of bounds");
  }
  const uint64_t nd = h.num_docs, nt = h.num_terms;
  if (h.section_len[S_DOC_LENS] != nd * 4) corrupt("doc_lens length");
  if (h.section_len[S_DOCID_OFFS] != (nd + 1) * 8) corrupt("docid offsets length");
  if (h.section_len[S_LEX_ENTRIES] != nt * sizeof(TermEntry)) corrupt("lexicon length");

  const auto* offs = reinterpret_cast<const uint64_t*>(base_ + h.section_off[S_DOCID_OFFS]);
  for (uint64_t d = 0; d < nd; ++d)
    if (offs[d] > offs[d + 1]) corrupt("docid offsets not monotone");
  if (offs[nd] != h.section_len[S_DOCID_BLOB]) corrupt("docid blob length");

  const auto* ents = reinterpret_cast<const TermEntry*>(base_ + h.section_off[S_LEX_ENTRIES]);
  const char* strs = reinterpret_cast<const char*>(base_ + h.section_off[S_LEX_STRINGS]);
  const uint8_t* post = base_ + h.section_off[S_POSTINGS];
  const uint64_t slen = h.section_len[S_LEX_STRINGS], plen = h.section_len[S_POSTINGS];
  std::string_view prev;
  uint64_t postings = 0;
  for (uint64_t t = 0; t < nt; ++t) {
    const TermEntry& e = ents[t];
    if (e.str_off > slen || e.str_len > slen - e.str_off) corrupt("term string out of bounds");
    std::string_view cur(strs + e.str_off, e.str_len);
    if (t > 0 && !(prev < cur)) corrupt("lexicon not strictly sorted");
    prev = cur;
    if (e.docs_off > plen || e.docs_len > plen - e.docs_off) corrupt("doc stream out of bounds");
    if (e.pos_off > plen || e.pos_len > plen - e.pos_off) corrupt("pos stream out of bounds");
    if (e.df == 0 || e.df > nd) corrupt("bad df");
    // Walk the doc stream with checked decoding: ids in range and strictly
    // increasing, tf >= 1, bytes used exactly, and the position stream must
    // hold exactly cf well-formed values. This is what makes the unchecked
    // hot-path decoder safe.
    const uint8_t* p = post + e.docs_off;
    const uint8_t* pend = p + e.docs_len;
    uint64_t doc = 0, cf = 0;
    for (uint32_t i = 0; i < e.df; ++i) {
      uint32_t gap, tf;
      if (!vbyte_decode_checked(p, pend, gap) || !vbyte_decode_checked(p, pend, tf))
        corrupt("truncated doc stream");
      if (i > 0 && gap == 0) corrupt("non-increasing doc ids");
      doc += gap;
      if (doc >= nd) corrupt("doc id out of range");
      if (tf == 0) corrupt("zero tf");
      cf += tf;
    }
    if (p != pend) corrupt("doc stream has trailing bytes");
    if (cf != e.cf) corrupt("cf mismatch");
    const uint8_t* q = post + e.pos_off;
    const uint8_t* qend = q + e.pos_len;
    for (uint64_t i = 0; i < cf; ++i) {
      uint32_t v;
      if (!vbyte_decode_checked(q, qend, v)) corrupt("truncated position stream");
    }
    if (q != qend) corrupt("position stream has trailing bytes");
    postings += e.df;
  }
  if (postings != h.num_postings) corrupt("posting count mismatch");
}

Index::Index(Index&& o) noexcept { *this = std::move(o); }

Index& Index::operator=(Index&& o) noexcept {
  if (this != &o) {
    if (map_) ::munmap(map_, size_);
    map_ = o.map_;
    size_ = o.size_;
    base_ = o.base_;
    h_ = o.h_;
    doc_lens_ = o.doc_lens_;
    docid_offs_ = o.docid_offs_;
    docid_blob_ = o.docid_blob_;
    entries_ = o.entries_;
    strings_ = o.strings_;
    post_ = o.post_;
    o.map_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

Index::~Index() {
  if (map_) ::munmap(map_, size_);
}

std::string_view Index::doc_id(uint32_t d) const {
  return {docid_blob_ + docid_offs_[d], static_cast<size_t>(docid_offs_[d + 1] - docid_offs_[d])};
}

TokenizerOptions Index::tokenizer_options() const {
  TokenizerOptions o;
  o.stem = (h_->flags & F_STEM) != 0;
  o.stopwords = (h_->flags & F_STOPWORDS) != 0;
  return o;
}

std::string_view Index::term(const TermEntry& e) const { return {strings_ + e.str_off, e.str_len}; }

const TermEntry* Index::find(std::string_view term) const {
  const TermEntry* lo = entries_;
  const TermEntry* hi = entries_ + h_->num_terms;
  auto it = std::lower_bound(lo, hi, term, [&](const TermEntry& e, std::string_view t) {
    return std::string_view(strings_ + e.str_off, e.str_len) < t;
  });
  if (it != hi && std::string_view(strings_ + it->str_off, it->str_len) == term) return it;
  return nullptr;
}

}  // namespace se
