#include "kvdb/btree.h"

#include <algorithm>
#include <cstring>

namespace kvdb {

namespace {
// Index m such that ents[0, m) holds roughly half of the bytes.
int byte_midpoint(const std::vector<node::Entry>& ents) {
  size_t total = 0;
  for (const auto& e : ents) total += node::cell_size(e.key.size(), e.value.size()) + node::kSlotSize;
  size_t acc = 0;
  int m = 0;
  const int n = static_cast<int>(ents.size());
  while (m < n && acc < total / 2) {
    acc += node::cell_size(ents[m].key.size(), ents[m].value.size()) + node::kSlotSize;
    ++m;
  }
  return m;
}

std::string encode_child(PageId id) {
  std::string s(4, '\0');
  store<uint32_t>(s.data(), id);
  return s;
}
}  // namespace

void BTree::format(BufferPool& pool) {
  char* m = pool.create(kMetaPageId);
  store<uint64_t>(m + Meta::kMagicOff, Meta::kMagic);
  store<uint32_t>(m + Meta::kVersionOff, Meta::kVersion);
  store<uint32_t>(m + Meta::kRootOff, 1);
  store<uint32_t>(m + Meta::kPageCountOff, 2);
  store<uint64_t>(m + Meta::kLsnOff, 0);
  store<uint32_t>(m + Meta::kHeightOff, 1);
  char* r = pool.create(1);
  node::init(r, node::Type::kLeaf, kNoPage);
  pool.unpin(1, true);
  pool.unpin(kMetaPageId, true);
}

PageId BTree::root() {
  PageGuard m = pin(kMetaPageId);
  return load<uint32_t>(m.data() + Meta::kRootOff);
}

uint32_t BTree::height() {
  PageGuard m = pin(kMetaPageId);
  return load<uint32_t>(m.data() + Meta::kHeightOff);
}

uint64_t BTree::checkpoint_lsn() {
  PageGuard m = pin(kMetaPageId);
  return load<uint64_t>(m.data() + Meta::kLsnOff);
}

void BTree::set_checkpoint_lsn(uint64_t lsn) {
  PageGuard m = pin(kMetaPageId);
  store<uint64_t>(m.data() + Meta::kLsnOff, lsn);
  m.mark_dirty();
}

// Pages are only ever appended; deletes never free pages in v0.
PageId BTree::allocate(PageGuard* out) {
  PageGuard m = pin(kMetaPageId);
  const PageId id = load<uint32_t>(m.data() + Meta::kPageCountOff);
  store<uint32_t>(m.data() + Meta::kPageCountOff, id + 1);
  m.mark_dirty();
  *out = PageGuard(&pool_, id, pool_.create(id));
  out->mark_dirty();
  return id;
}

bool BTree::get(std::string_view key, std::string* out) {
  PageGuard g = pin(root());
  while (!node::is_leaf(g.data())) {
    const PageId c = node::find_child(g.data(), key);
    g = pin(c);  // move-assign unpins the parent first
  }
  const int i = node::lower_bound(g.data(), key);
  if (i < node::count(g.data()) && node::key(g.data(), i) == key) {
    if (out != nullptr) out->assign(node::value(g.data(), i));
    return true;
  }
  return false;
}

void BTree::put(std::string_view key, std::string_view value) {
  std::vector<PageId> path;  // internal nodes from root down
  PageGuard g = pin(root());
  while (!node::is_leaf(g.data())) {
    path.push_back(g.id());
    const PageId c = node::find_child(g.data(), key);
    g = pin(c);
  }
  char* p = g.data();
  const int idx = node::lower_bound(p, key);
  const bool exists = idx < node::count(p) && node::key(p, idx) == key;
  g.mark_dirty();
  if (exists) {
    if (node::update(p, idx, value)) return;
    auto ents = node::entries(p);
    ents[idx].value.assign(value);
    split_leaf(g, ents, -1, path);
    return;
  }
  if (node::insert(p, idx, key, value)) return;
  auto ents = node::entries(p);
  ents.insert(ents.begin() + idx, {std::string(key), std::string(value)});
  split_leaf(g, ents, idx, path);
}

// ents already contains the new entry. idx is where it landed, or -1 for an
// update; it is used only to detect appends to the rightmost leaf.
void BTree::split_leaf(PageGuard& leaf, std::vector<node::Entry>& ents, int idx,
                       std::vector<PageId>& path) {
  const int n = static_cast<int>(ents.size());
  int m;
  if (idx == n - 1 && node::link(leaf.data()) == kNoPage) {
    // Appending past the end of the rightmost leaf (ascending inserts): leave
    // the old leaf full and start a new one. Gives ~100% fill instead of 50%.
    m = n - 1;
  } else {
    m = std::clamp(byte_midpoint(ents), 1, n - 1);
  }
  PageGuard right;
  const PageId right_id = allocate(&right);
  node::build(right.data(), node::Type::kLeaf, node::link(leaf.data()),
              ents.data() + m, ents.data() + n);
  node::build(leaf.data(), node::Type::kLeaf, right_id, ents.data(), ents.data() + m);
  leaf.mark_dirty();
  const PageId left_id = leaf.id();
  std::string sep = ents[m].key;
  right.release();
  leaf.release();
  insert_into_parent(path, left_id, std::move(sep), right_id);
}

void BTree::insert_into_parent(std::vector<PageId>& path, PageId left,
                               std::string sep, PageId right) {
  if (path.empty()) {  // split the root: tree grows by one level
    PageGuard r;
    const PageId new_root = allocate(&r);
    node::init(r.data(), node::Type::kInternal, left);
    node::insert(r.data(), 0, sep, encode_child(right));
    PageGuard m = pin(kMetaPageId);
    store<uint32_t>(m.data() + Meta::kRootOff, new_root);
    store<uint32_t>(m.data() + Meta::kHeightOff,
                    load<uint32_t>(m.data() + Meta::kHeightOff) + 1);
    m.mark_dirty();
    return;
  }
  const PageId pid = path.back();
  path.pop_back();
  PageGuard pg = pin(pid);
  pg.mark_dirty();
  char* p = pg.data();
  const int idx = node::lower_bound(p, sep);
  const std::string cv = encode_child(right);
  if (node::insert(p, idx, sep, cv)) return;

  // Split the internal node. The middle key moves up; its child becomes the
  // new right node's leftmost child.
  auto ents = node::entries(p);
  ents.insert(ents.begin() + idx, {std::move(sep), cv});
  const int n = static_cast<int>(ents.size());
  // Split by bytes, not count: keys vary in size. Both sides keep >= 1 key.
  const int mid = std::clamp(byte_midpoint(ents), 1, n - 2);
  PageGuard rg;
  const PageId rid = allocate(&rg);
  const PageId mid_child = load<uint32_t>(ents[mid].value.data());
  node::build(rg.data(), node::Type::kInternal, mid_child, ents.data() + mid + 1,
              ents.data() + n);
  node::build(p, node::Type::kInternal, node::link(p), ents.data(), ents.data() + mid);
  std::string up = std::move(ents[mid].key);
  rg.release();
  pg.release();
  insert_into_parent(path, pid, std::move(up), rid);
}

bool BTree::del(std::string_view key) {
  PageGuard g = pin(root());
  while (!node::is_leaf(g.data())) {
    const PageId c = node::find_child(g.data(), key);
    g = pin(c);
  }
  const int i = node::lower_bound(g.data(), key);
  if (i < node::count(g.data()) && node::key(g.data(), i) == key) {
    node::erase(g.data(), i);
    g.mark_dirty();
    return true;
  }
  return false;
}

void BTree::scan(std::string_view lo, std::string_view hi,
                 const std::function<bool(std::string_view, std::string_view)>& fn) {
  PageGuard g = pin(root());
  while (!node::is_leaf(g.data())) {
    const PageId c = node::find_child(g.data(), lo);
    g = pin(c);
  }
  int i = node::lower_bound(g.data(), lo);
  for (;;) {
    const char* p = g.data();
    for (int n = node::count(p); i < n; ++i) {
      const auto k = node::key(p, i);
      if (!hi.empty() && k >= hi) return;
      if (!fn(k, node::value(p, i))) return;
    }
    const PageId next = node::link(p);
    if (next == kNoPage) return;
    g = pin(next);
    i = 0;
  }
}

TreeStats BTree::verify() {
  TreeStats st;
  st.height = height();
  size_t leaf_bytes = 0;
  std::string why;
  // Depth-first walk carrying the key range [lo, hi) each subtree must obey.
  struct Item {
    PageId id;
    std::string lo, hi;
    bool has_lo, has_hi;
    uint32_t depth;
  };
  std::vector<Item> stack{{root(), "", "", false, false, 1}};
  std::vector<PageId> leaves_in_order;
  while (!stack.empty()) {
    Item it = std::move(stack.back());
    stack.pop_back();
    PageGuard g = pin(it.id);
    const char* p = g.data();
    if (!node::check(p, &why)) throw std::logic_error("page " + std::to_string(it.id) + ": " + why);
    const int n = node::count(p);
    for (int i = 0; i < n; ++i) {
      const auto k = node::key(p, i);
      if ((it.has_lo && k < it.lo) || (it.has_hi && k >= it.hi)) {
        throw std::logic_error("key out of parent range in page " + std::to_string(it.id));
      }
    }
    if (node::is_leaf(p)) {
      if (it.depth != st.height) throw std::logic_error("leaf at wrong depth");
      ++st.leaf_pages;
      st.entries += n;
      leaf_bytes += node::used_bytes(p);
      leaves_in_order.push_back(it.id);
      continue;
    }
    ++st.internal_pages;
    if (n == 0) throw std::logic_error("internal node with no keys");
    // Push children right-to-left so leaves are visited left-to-right.
    for (int i = n - 1; i >= -1; --i) {
      Item c;
      c.id = i < 0 ? node::link(p) : node::child(p, i);
      c.depth = it.depth + 1;
      if (i < 0) { c.lo = it.lo; c.has_lo = it.has_lo; }
      else { c.lo = std::string(node::key(p, i)); c.has_lo = true; }
      if (i + 1 < n) { c.hi = std::string(node::key(p, i + 1)); c.has_hi = true; }
      else { c.hi = it.hi; c.has_hi = it.has_hi; }
      stack.push_back(std::move(c));
    }
  }
  for (size_t i = 0; i < leaves_in_order.size(); ++i) {
    PageGuard g = pin(leaves_in_order[i]);
    const PageId expect = i + 1 < leaves_in_order.size() ? leaves_in_order[i + 1] : kNoPage;
    if (node::link(g.data()) != expect) throw std::logic_error("broken leaf sibling chain");
  }
  st.leaf_fill = st.leaf_pages ? double(leaf_bytes) / double(st.leaf_pages * node::kUsable) : 0;
  return st;
}

}  // namespace kvdb
