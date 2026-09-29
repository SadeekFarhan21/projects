#include "kvdb/db.h"

#include <cerrno>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace kvdb {

namespace {

constexpr uint64_t kJournalMagic = 0x4C4E524A42445653ull;  // "SVDBJRNL"
constexpr size_t kJournalHeader = 16;  // magic u64, count u32, crc u32
constexpr size_t kJournalEntry = 4 + kPageSize;
// Frames kept free of dirty pages so one operation can always make progress:
// a put dirties at most 2 pages per level plus a new root and the meta page.
constexpr size_t kPoolReserve = 64;

void fsync_parent_dir(const std::string& path) {
  std::string copy = path;
  const char* dir = ::dirname(copy.data());
  int fd = ::open(dir, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return;  // best effort
  ::fsync(fd);
  ::close(fd);
}

void pwrite_all(int fd, const char* p, size_t n, off_t off) {
  size_t done = 0;
  while (done < n) {
    ssize_t w = ::pwrite(fd, p + done, n - done, off + static_cast<off_t>(done));
    if (w < 0) {
      if (errno == EINTR) continue;
      throw_errno("pwrite journal");
    }
    done += static_cast<size_t>(w);
  }
}

}  // namespace

std::unique_ptr<DB> DB::open(const std::string& path, const Options& opts) {
  std::unique_ptr<DB> db(new DB(path, opts));
  db->recover();
  return db;
}

DB::DB(const std::string& path, const Options& opts) : path_(path), opts_(opts) {
  if (opts_.pool_pages < 4 * kPoolReserve) {
    throw std::invalid_argument("pool_pages must be >= 256");
  }
  pager_ = std::make_unique<Pager>(path);  // takes the file lock first
  journal_fd_ = ::open((path + "-journal").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (journal_fd_ < 0) throw_errno("open journal");
  wal_ = std::make_unique<Wal>(path + "-wal");
  fsync_parent_dir(path);  // make the three directory entries durable
}

DB::~DB() {
  try {
    if (tree_) checkpoint_impl(true);  // null if open() failed midway
  } catch (...) {
    // Destructors must not throw; the WAL still has everything.
  }
  if (journal_fd_ >= 0) ::close(journal_fd_);
}

// Startup: (1) finish or discard an interrupted checkpoint, (2) load the
// tree as of the last checkpoint, (3) redo WAL records newer than it,
// (4) checkpoint so the WAL starts empty.
void DB::recover() {
  recover_journal();
  pool_ = std::make_unique<BufferPool>(*pager_, opts_.pool_pages);
  tree_ = std::make_unique<BTree>(*pool_);
  if (pager_->file_pages() == 0) {
    BTree::format(*pool_);  // brand-new database, only in memory for now
  } else {
    PageGuard m(pool_.get(), kMetaPageId, pool_->fetch(kMetaPageId));
    if (load<uint64_t>(m.data() + Meta::kMagicOff) != Meta::kMagic) {
      throw IoError("not a kvdb file (bad magic): " + path_);
    }
  }
  const uint64_t ckpt_lsn = tree_->checkpoint_lsn();
  applied_lsn_ = ckpt_lsn;
  const uint64_t last = wal_->replay([&](uint64_t lsn, std::string_view rep) {
    if (lsn <= ckpt_lsn) return;  // already reflected in the page file
    apply(rep);
    applied_lsn_ = lsn;
    ++stats_.recovered_records;
    // Never truncate the WAL mid-replay: its remaining records are needed.
    maybe_checkpoint(false);
  });
  next_lsn_ = std::max(ckpt_lsn, last) + 1;
  applied_lsn_ = next_lsn_ - 1;
  checkpoint_impl(true);
}

// A journal is valid only if it is complete (checksum over every byte). A
// valid journal means the crash happened after it was fsynced, possibly in
// the middle of the in-place writes, so we redo all of them. An invalid one
// means the crash happened before any in-place write, so we ignore it.
void DB::recover_journal() {
  struct stat st {};
  if (::fstat(journal_fd_, &st) != 0) throw_errno("fstat journal");
  const size_t size = static_cast<size_t>(st.st_size);
  if (size >= kJournalHeader) {
    std::string data(size, '\0');
    if (::pread(journal_fd_, data.data(), size, 0) != static_cast<ssize_t>(size)) {
      throw_errno("pread journal");
    }
    const uint64_t magic = load<uint64_t>(data.data());
    const uint32_t count = load<uint32_t>(data.data() + 8);
    const uint32_t crc = load<uint32_t>(data.data() + 12);
    const bool valid = magic == kJournalMagic &&
                       size == kJournalHeader + size_t{count} * kJournalEntry &&
                       crc32(data.data() + kJournalHeader, size - kJournalHeader) == crc;
    if (valid) {
      for (uint32_t i = 0; i < count; ++i) {
        const char* e = data.data() + kJournalHeader + size_t{i} * kJournalEntry;
        pager_->write_page(load<uint32_t>(e), e + 4);
        ++stats_.journal_pages_replayed;
      }
      pager_->sync(opts_.sync == SyncMode::kNone ? SyncMode::kFsync : opts_.sync);
    }
  }
  if (size != 0) {
    if (::ftruncate(journal_fd_, 0) != 0) throw_errno("ftruncate journal");
    sync_fd(journal_fd_, SyncMode::kFsync);
  }
}

void DB::apply(std::string_view rep) {
  const bool ok = WriteBatch::iterate(rep, [&](WriteBatch::Kind kind, std::string_view k,
                                               std::string_view v) {
    if (kind == WriteBatch::kPut) tree_->put(k, v);
    else tree_->del(k);
    // A large batch can dirty more pages than the pool holds. Checkpointing
    // here is safe because checkpoint records applied_lsn_ (the previous
    // batch) and keeps the WAL, so this batch is redone in full on recovery.
    if (pool_->dirty_count() + kPoolReserve >= pool_->capacity()) checkpoint_impl(false);
  });
  if (!ok) throw IoError("malformed write batch");
}

void DB::maybe_checkpoint(bool between_batches) {
  const bool pool_pressure = pool_->dirty_count() + kPoolReserve >= pool_->capacity();
  const bool wal_big = between_batches && wal_->size() >= opts_.wal_checkpoint_bytes;
  if (pool_pressure || wal_big) checkpoint_impl(between_batches);
}

void DB::write(const WriteBatch& batch) {
  if (batch.count() == 0) return;
  maybe_checkpoint(true);
  const uint64_t lsn = next_lsn_++;
  wal_->append(lsn, batch.rep(), opts_.sync);  // commit point
  apply(batch.rep());
  applied_lsn_ = lsn;
  ++stats_.commits;
}

void DB::put(std::string_view key, std::string_view value) {
  scratch_.clear();
  scratch_.put(key, value);
  write(scratch_);
}

bool DB::del(std::string_view key) {
  // Deleting a missing key is not logged at all.
  if (!tree_->get(key, nullptr)) return false;
  scratch_.clear();
  scratch_.del(key);
  write(scratch_);
  return true;
}

bool DB::get(std::string_view key, std::string* value) { return tree_->get(key, value); }

void DB::scan(std::string_view lo, std::string_view hi,
              const std::function<bool(std::string_view, std::string_view)>& fn) {
  tree_->scan(lo, hi, fn);
}

// Checkpoint protocol (see DESIGN.md):
//   1. stamp meta with applied_lsn_; collect dirty pages (meta included)
//   2. write them all to the journal, header last, fsync   <- atomicity point
//   3. write them in place in the page file, fsync
//   4. optionally truncate the WAL; truncate the journal
void DB::checkpoint_impl(bool truncate_wal) {
  using Clock = std::chrono::steady_clock;
  auto us_since = [](Clock::time_point t) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t).count());
  };
  auto t = Clock::now();
  tree_->set_checkpoint_lsn(applied_lsn_);
  const auto pages = pool_->dirty_pages();
  const SyncMode mode = opts_.sync;
  if (!pages.empty()) {
    // Stream the body, then write the header (count + CRC) at offset 0.
    std::vector<char> chunk;
    chunk.reserve(256 * kJournalEntry);
    uint32_t crc = 0;
    off_t off = kJournalHeader;
    auto flush_chunk = [&]() {
      crc = crc32(chunk.data(), chunk.size(), crc);
      pwrite_all(journal_fd_, chunk.data(), chunk.size(), off);
      off += static_cast<off_t>(chunk.size());
      chunk.clear();
    };
    for (const auto& [id, data] : pages) {
      char idb[4];
      store<uint32_t>(idb, id);
      chunk.insert(chunk.end(), idb, idb + 4);
      chunk.insert(chunk.end(), data, data + kPageSize);
      if (chunk.size() >= 256 * kJournalEntry) flush_chunk();
    }
    if (!chunk.empty()) flush_chunk();
    char hdr[kJournalHeader];
    store<uint64_t>(hdr, kJournalMagic);
    store<uint32_t>(hdr + 8, static_cast<uint32_t>(pages.size()));
    store<uint32_t>(hdr + 12, crc);
    failpoint::hit("ckpt.before_journal_header");
    pwrite_all(journal_fd_, hdr, kJournalHeader, 0);
    sync_fd(journal_fd_, mode);
    failpoint::hit("ckpt.journal_synced");
    stats_.ckpt_journal_us += us_since(t);
    t = Clock::now();

    for (size_t i = 0; i < pages.size(); ++i) {
      pager_->write_page(pages[i].first, pages[i].second);
      if (i == pages.size() / 2) failpoint::hit("ckpt.mid_data");
    }
    pager_->sync(mode);
    failpoint::hit("ckpt.data_synced");
    stats_.checkpoint_pages += pages.size();
    stats_.ckpt_data_us += us_since(t);
    t = Clock::now();
  }
  if (truncate_wal && wal_->size() != 0) {
    wal_->reset(mode);
    failpoint::hit("ckpt.wal_reset");
  }
  if (!pages.empty()) {
    if (::ftruncate(journal_fd_, 0) != 0) throw_errno("ftruncate journal");
    sync_fd(journal_fd_, mode);
  }
  pool_->mark_all_clean();
  stats_.ckpt_tail_us += us_since(t);
  ++stats_.checkpoints;
}

DbStats DB::stats() const {
  DbStats s = stats_;
  s.wal_bytes = wal_->bytes_appended();
  s.wal_syncs = wal_->syncs();
  s.pool = pool_->stats();
  s.page_reads = pager_->reads();
  s.page_writes = pager_->writes();
  return s;
}

}  // namespace kvdb
