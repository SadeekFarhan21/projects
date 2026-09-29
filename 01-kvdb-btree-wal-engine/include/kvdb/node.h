// Slotted-page layout for B+ tree nodes.
//
//   0      1      2        4        8           10     12      16
//   +------+------+--------+--------+-----------+------+-------+
//   | type | rsvd | nslots |  link  | cellStart | frag | rsvd  |
//   +------+------+--------+--------+-----------+------+-------+
//   | slot[0] slot[1] ... (u16 offsets, sorted by key) -->     |
//   |                     free space                           |
//   |           <-- cells (klen u16, vlen u16, key, value)     |
//   +----------------------------------------------------------+
//
// Leaf:     link = right sibling leaf (0 = none); value = user value.
// Internal: link = leftmost child; value = 4-byte child page id. Child i holds
//           keys >= key[i]; the leftmost child holds keys < key[0].
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "kvdb/common.h"

namespace kvdb::node {

enum class Type : uint8_t { kLeaf = 1, kInternal = 2 };

inline constexpr size_t kHeaderSize = 16;
inline constexpr size_t kSlotSize = 2;
inline constexpr size_t kCellHeader = 4;
inline constexpr size_t kUsable = kPageSize - kHeaderSize;

struct Entry {
  std::string key;
  std::string value;
};

inline size_t cell_size(size_t klen, size_t vlen) {
  return kCellHeader + klen + vlen;
}

void init(char* p, Type t, PageId link);
inline Type type(const char* p) { return static_cast<Type>(p[0]); }
inline bool is_leaf(const char* p) { return type(p) == Type::kLeaf; }
inline uint16_t count(const char* p) { return load<uint16_t>(p + 2); }
inline PageId link(const char* p) { return load<uint32_t>(p + 4); }
inline void set_link(char* p, PageId id) { store<uint32_t>(p + 4, id); }

std::string_view key(const char* p, int i);
std::string_view value(const char* p, int i);
PageId child(const char* p, int i);

// First slot whose key is >= k (or count() if none).
int lower_bound(const char* p, std::string_view k);
// Child page to follow for key k in an internal node.
PageId find_child(const char* p, std::string_view k);

// Inserts a cell at slot idx, compacting first if fragmented. Returns false
// (page unchanged) if it does not fit even after compaction.
bool insert(char* p, int idx, std::string_view k, std::string_view v);
// Overwrites the value at idx if the new value fits; returns false otherwise.
bool update(char* p, int idx, std::string_view v);
void erase(char* p, int idx);

// Bytes of live cells plus their slots.
size_t used_bytes(const char* p);

std::vector<Entry> entries(const char* p);
void build(char* p, Type t, PageId link, const Entry* begin, const Entry* end);

// Structural self-check used by tests: sorted keys, sane offsets.
bool check(const char* p, std::string* why);

}  // namespace kvdb::node
