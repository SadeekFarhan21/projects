#include "se/index_builder.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "se/format.hpp"
#include "se/varint.hpp"

namespace se {

std::vector<Token> IndexBuilder::tokenize_document(const Tokenizer& tok, std::string_view title,
                                                   std::string_view text) {
  std::vector<Token> toks;
  uint32_t next = tok.tokenize(title, toks, 0);
  if (next > 0) ++next;  // one-position gap between fields
  tok.tokenize(text, toks, next);
  return toks;
}

uint32_t IndexBuilder::add_document(std::string_view ext_id, std::string_view title,
                                    std::string_view text) {
  return add_tokenized(ext_id, tokenize_document(tok_, title, text));
}

uint32_t IndexBuilder::add_tokenized(std::string_view ext_id, const std::vector<Token>& tokens) {
  const uint32_t doc = num_docs();
  // Map terms to ids, then sort (term id, pos) so each term's positions are
  // contiguous and ascending. Positions arrive ascending, so a stable sort by
  // term id alone would do; sorting the pair is just as cheap and clearer.
  scratch_.clear();
  scratch_.reserve(tokens.size());
  for (const Token& t : tokens) {
    auto it = term_ids_.find(std::string_view(t.term));
    uint32_t tid;
    if (it == term_ids_.end()) {
      tid = static_cast<uint32_t>(terms_.size());
      term_ids_.emplace(t.term, tid);
      terms_.push_back(t.term);
      postings_.emplace_back();
    } else {
      tid = it->second;
    }
    scratch_.emplace_back(tid, t.pos);
  }
  std::sort(scratch_.begin(), scratch_.end());

  for (size_t i = 0; i < scratch_.size();) {
    uint32_t tid = scratch_[i].first;
    size_t e = i;
    while (e < scratch_.size() && scratch_[e].first == tid) ++e;
    TermPostings& tp = postings_[tid];
    uint32_t tf = static_cast<uint32_t>(e - i);
    vbyte_encode(doc - tp.last_doc, tp.docs);
    vbyte_encode(tf, tp.docs);
    uint32_t prev = 0;
    for (size_t k = i; k < e; ++k) {
      vbyte_encode(scratch_[k].second - prev, tp.pos);
      prev = scratch_[k].second;
    }
    tp.last_doc = doc;
    tp.df += 1;
    tp.cf += tf;
    i = e;
  }

  doc_lens_.push_back(static_cast<uint32_t>(tokens.size()));
  total_len_ += tokens.size();
  ext_ids_.emplace_back(ext_id);
  return doc;
}

namespace {

void pad8(std::vector<uint8_t>& buf) {
  while (buf.size() % 8) buf.push_back(0);
}

template <typename T>
void append_pod(std::vector<uint8_t>& buf, const T* data, size_t count) {
  size_t off = buf.size();
  buf.resize(off + sizeof(T) * count);
  if (count) std::memcpy(buf.data() + off, data, sizeof(T) * count);
}

}  // namespace

BuildStats IndexBuilder::write(const std::string& path) const {
  BuildStats st;
  Header h{};
  std::memcpy(h.magic, kMagic, sizeof(kMagic));
  h.version = kFormatVersion;
  h.flags = (tok_.options().stem ? F_STEM : 0u) | (tok_.options().stopwords ? F_STOPWORDS : 0u);
  h.num_docs = num_docs();
  h.num_terms = static_cast<uint32_t>(terms_.size());
  h.total_doc_len = total_len_;

  // Lexicographic term order enables binary-search lookup with no load step.
  std::vector<uint32_t> order(terms_.size());
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(),
            [&](uint32_t a, uint32_t b) { return terms_[a] < terms_[b]; });

  std::vector<uint8_t> buf;
  buf.resize(sizeof(Header));
  pad8(buf);

  auto begin_section = [&](Section s) {
    pad8(buf);
    h.section_off[s] = buf.size();
  };
  auto end_section = [&](Section s) { h.section_len[s] = buf.size() - h.section_off[s]; };

  begin_section(S_DOC_LENS);
  append_pod(buf, doc_lens_.data(), doc_lens_.size());
  end_section(S_DOC_LENS);

  begin_section(S_DOCID_OFFS);
  {
    std::vector<uint64_t> offs;
    offs.reserve(ext_ids_.size() + 1);
    uint64_t o = 0;
    for (const auto& id : ext_ids_) {
      offs.push_back(o);
      o += id.size();
    }
    offs.push_back(o);
    append_pod(buf, offs.data(), offs.size());
  }
  end_section(S_DOCID_OFFS);

  begin_section(S_DOCID_BLOB);
  for (const auto& id : ext_ids_) append_pod(buf, id.data(), id.size());
  end_section(S_DOCID_BLOB);

  // Lay out strings and postings first so the entries can carry offsets.
  std::vector<TermEntry> entries(order.size());
  std::vector<uint8_t> strings, post;
  for (size_t r = 0; r < order.size(); ++r) {
    const uint32_t tid = order[r];
    const TermPostings& tp = postings_[tid];
    TermEntry& e = entries[r];
    e.str_off = strings.size();
    e.str_len = static_cast<uint32_t>(terms_[tid].size());
    strings.insert(strings.end(), terms_[tid].begin(), terms_[tid].end());
    e.df = tp.df;
    e.cf = tp.cf;
    e.docs_off = post.size();
    e.docs_len = static_cast<uint32_t>(tp.docs.size());
    post.insert(post.end(), tp.docs.begin(), tp.docs.end());
    e.pos_off = post.size();
    e.pos_len = static_cast<uint32_t>(tp.pos.size());
    post.insert(post.end(), tp.pos.begin(), tp.pos.end());
    h.num_postings += tp.df;
    h.num_positions += tp.cf;
    st.doc_stream_bytes += tp.docs.size();
    st.pos_stream_bytes += tp.pos.size();
  }

  begin_section(S_LEX_ENTRIES);
  append_pod(buf, entries.data(), entries.size());
  end_section(S_LEX_ENTRIES);

  begin_section(S_LEX_STRINGS);
  append_pod(buf, strings.data(), strings.size());
  end_section(S_LEX_STRINGS);

  begin_section(S_POSTINGS);
  append_pod(buf, post.data(), post.size());
  end_section(S_POSTINGS);

  pad8(buf);
  h.file_size = buf.size();
  std::memcpy(buf.data(), &h, sizeof(h));

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open " + path + " for writing");
  out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  if (!out) throw std::runtime_error("write failed: " + path);

  st.num_docs = h.num_docs;
  st.num_terms = h.num_terms;
  st.num_postings = h.num_postings;
  st.num_positions = h.num_positions;
  st.file_bytes = h.file_size;
  for (int s = 0; s < S_COUNT; ++s) st.section_bytes[s] = h.section_len[s];
  st.raw_doc_stream_bytes = h.num_postings * 8;
  st.raw_pos_stream_bytes = h.num_positions * 4;
  return st;
}

}  // namespace se
