#include "kvdb/node.h"

#include <algorithm>
#include <cstring>

namespace kvdb::node {
namespace {

uint16_t cell_start(const char* p) { return load<uint16_t>(p + 8); }
void set_cell_start(char* p, uint16_t v) { store<uint16_t>(p + 8, v); }
uint16_t frag(const char* p) { return load<uint16_t>(p + 10); }
void set_frag(char* p, uint16_t v) { store<uint16_t>(p + 10, v); }
void set_count(char* p, uint16_t n) { store<uint16_t>(p + 2, n); }
uint16_t slot(const char* p, int i) {
  return load<uint16_t>(p + kHeaderSize + kSlotSize * i);
}
void set_slot(char* p, int i, uint16_t off) {
  store<uint16_t>(p + kHeaderSize + kSlotSize * i, off);
}
size_t contiguous_free(const char* p) {
  return cell_start(p) - (kHeaderSize + kSlotSize * count(p));
}
size_t cell_size_at(const char* p, int i) {
  const char* c = p + slot(p, i);
  return cell_size(load<uint16_t>(c), load<uint16_t>(c + 2));
}

// Rewrites the page with cells packed at the end, reclaiming dead bytes.
void compact(char* p) {
  char tmp[kPageSize];
  std::memcpy(tmp, p, kPageSize);
  const int n = count(p);
  uint16_t top = kPageSize;
  for (int i = 0; i < n; ++i) {
    const size_t sz = cell_size_at(tmp, i);
    top = static_cast<uint16_t>(top - sz);
    std::memcpy(p + top, tmp + slot(tmp, i), sz);
    set_slot(p, i, top);
  }
  set_cell_start(p, top);
  set_frag(p, 0);
}

}  // namespace

void init(char* p, Type t, PageId link) {
  std::memset(p, 0, kHeaderSize);
  p[0] = static_cast<char>(t);
  set_count(p, 0);
  set_link(p, link);
  set_cell_start(p, static_cast<uint16_t>(kPageSize));
  set_frag(p, 0);
}

std::string_view key(const char* p, int i) {
  const char* c = p + slot(p, i);
  return {c + kCellHeader, load<uint16_t>(c)};
}

std::string_view value(const char* p, int i) {
  const char* c = p + slot(p, i);
  const uint16_t klen = load<uint16_t>(c);
  return {c + kCellHeader + klen, load<uint16_t>(c + 2)};
}

PageId child(const char* p, int i) { return load<uint32_t>(value(p, i).data()); }

int lower_bound(const char* p, std::string_view k) {
  int lo = 0, hi = count(p);
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (key(p, mid) < k) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

PageId find_child(const char* p, std::string_view k) {
  // Last separator <= k; none means the leftmost child.
  int lo = 0, hi = count(p);
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (key(p, mid) <= k) lo = mid + 1;
    else hi = mid;
  }
  return lo == 0 ? link(p) : child(p, lo - 1);
}

size_t used_bytes(const char* p) {
  size_t total = 0;
  for (int i = 0, n = count(p); i < n; ++i) total += cell_size_at(p, i) + kSlotSize;
  return total;
}

bool insert(char* p, int idx, std::string_view k, std::string_view v) {
  const size_t need = cell_size(k.size(), v.size());
  if (contiguous_free(p) < need + kSlotSize) {
    if (contiguous_free(p) + frag(p) < need + kSlotSize) return false;
    compact(p);
  }
  const int n = count(p);
  const uint16_t off = static_cast<uint16_t>(cell_start(p) - need);
  char* c = p + off;
  store<uint16_t>(c, static_cast<uint16_t>(k.size()));
  store<uint16_t>(c + 2, static_cast<uint16_t>(v.size()));
  std::memcpy(c + kCellHeader, k.data(), k.size());
  std::memcpy(c + kCellHeader + k.size(), v.data(), v.size());
  char* slots = p + kHeaderSize;
  std::memmove(slots + kSlotSize * (idx + 1), slots + kSlotSize * idx,
               kSlotSize * (n - idx));
  set_slot(p, idx, off);
  set_count(p, static_cast<uint16_t>(n + 1));
  set_cell_start(p, off);
  return true;
}

bool update(char* p, int idx, std::string_view v) {
  char* c = p + slot(p, idx);
  const uint16_t klen = load<uint16_t>(c);
  const uint16_t vlen = load<uint16_t>(c + 2);
  if (v.size() == vlen) {  // common case: same-size overwrite in place
    std::memcpy(c + kCellHeader + klen, v.data(), v.size());
    return true;
  }
  const std::string k(c + kCellHeader, klen);  // copy: erase may move bytes
  // Would it fit once the old cell is gone? Check before mutating anything.
  const size_t free_after = contiguous_free(p) + frag(p) + cell_size(klen, vlen);
  if (free_after < cell_size(klen, v.size())) return false;
  erase(p, idx);
  const bool ok = insert(p, idx, k, v);
  (void)ok;  // guaranteed by the check above
  return true;
}

void erase(char* p, int idx) {
  const int n = count(p);
  const size_t sz = cell_size_at(p, idx);
  const uint16_t off = slot(p, idx);
  char* slots = p + kHeaderSize;
  std::memmove(slots + kSlotSize * idx, slots + kSlotSize * (idx + 1),
               kSlotSize * (n - idx - 1));
  set_count(p, static_cast<uint16_t>(n - 1));
  if (off == cell_start(p)) {
    set_cell_start(p, static_cast<uint16_t>(off + sz));  // lowest cell: reclaim
  } else {
    set_frag(p, static_cast<uint16_t>(frag(p) + sz));
  }
}

std::vector<Entry> entries(const char* p) {
  std::vector<Entry> out;
  const int n = count(p);
  out.reserve(n + 1);
  for (int i = 0; i < n; ++i) out.push_back({std::string(key(p, i)), std::string(value(p, i))});
  return out;
}

void build(char* p, Type t, PageId link, const Entry* begin, const Entry* end) {
  init(p, t, link);
  int i = 0;
  for (const Entry* e = begin; e != end; ++e, ++i) {
    if (!insert(p, i, e->key, e->value)) throw std::logic_error("node::build overflow");
  }
}

bool check(const char* p, std::string* why) {
  if (type(p) != Type::kLeaf && type(p) != Type::kInternal) {
    *why = "bad node type";
    return false;
  }
  const int n = count(p);
  if (kHeaderSize + kSlotSize * n > cell_start(p) || cell_start(p) > kPageSize) {
    *why = "slot array overlaps cells";
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (slot(p, i) < cell_start(p) || slot(p, i) + cell_size_at(p, i) > kPageSize) {
      *why = "cell out of bounds";
      return false;
    }
    if (i > 0 && !(key(p, i - 1) < key(p, i))) {
      *why = "keys not strictly increasing";
      return false;
    }
  }
  return true;
}

}  // namespace kvdb::node
