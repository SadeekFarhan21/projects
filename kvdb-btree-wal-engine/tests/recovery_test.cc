// Crash-recovery tests. A child process does the writes and "crashes" either
// by SIGKILL from the parent at a random moment, or by _exit() at a named
// step inside checkpoint (failpoints). The parent then reopens the database
// and checks it against a model of every acknowledged operation.
#include <gtest/gtest.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <map>
#include <random>

#include "kvdb/db.h"
#include "test_util.h"

using namespace kvdb;
using test::key_of;
using test::value_of;

namespace {

using Model = std::map<std::string, std::string>;

// Operation j of a deterministic workload: mostly single puts, some deletes,
// and every 25th op is a 20-op batch (to check batch atomicity).
WriteBatch gen_op(uint64_t j) {
  std::mt19937_64 rng(j * 0x9E3779B97F4A7C15ull + 1);
  WriteBatch b;
  const int n = (j % 25 == 24) ? 20 : 1;
  for (int i = 0; i < n; ++i) {
    const uint64_t k = rng() % 4000;
    if (rng() % 5 == 0) b.del(key_of(k));
    else b.put(key_of(k), value_of(k, j));
  }
  return b;
}

void apply_op(Model& m, const WriteBatch& b) {
  WriteBatch::iterate(b.rep(), [&](WriteBatch::Kind kind, std::string_view k, std::string_view v) {
    if (kind == WriteBatch::kPut) m[std::string(k)] = std::string(v);
    else m.erase(std::string(k));
  });
}

Model dump(DB& db) {
  Model m;
  db.scan("", "", [&](std::string_view k, std::string_view v) {
    m.emplace(std::string(k), std::string(v));
    return true;
  });
  return m;
}

Options crashy_opts(SyncMode sync = SyncMode::kNone) {
  Options o;
  o.sync = sync;
  o.pool_pages = 256;                  // tiny pool: frequent pool-pressure checkpoints
  o.wal_checkpoint_bytes = 64 << 10;   // and frequent WAL-size checkpoints
  return o;
}

int wait_child(pid_t pid) {
  int status = 0;
  EXPECT_EQ(::waitpid(pid, &status, 0), pid);
  return status;
}

// One round: child applies ops [start, ...) forever, acking each op index on
// a pipe after write() returns. Parent kills it after `kill_after` acks.
// Returns the index of the last acknowledged op.
int64_t run_and_kill(const std::string& path, uint64_t start, int kill_after, SyncMode sync) {
  int fds[2];
  EXPECT_EQ(::pipe(fds), 0);
  pid_t pid = ::fork();
  if (pid == 0) {
    ::close(fds[0]);
    try {
      auto db = DB::open(path, crashy_opts(sync));
      for (uint64_t j = start;; ++j) {
        db->write(gen_op(j));
        const int64_t ack = static_cast<int64_t>(j);
        if (::write(fds[1], &ack, sizeof ack) != sizeof ack) ::_exit(3);
      }
    } catch (...) {
      ::_exit(4);
    }
  }
  ::close(fds[1]);
  int64_t last = static_cast<int64_t>(start) - 1, ack = 0;
  int seen = 0;
  while (seen < kill_after && ::read(fds[0], &ack, sizeof ack) == sizeof ack) {
    last = ack;
    ++seen;
  }
  ::kill(pid, SIGKILL);
  // Drain acks the child managed to send before dying: those were committed.
  while (::read(fds[0], &ack, sizeof ack) == sizeof ack) last = ack;
  ::close(fds[0]);
  const int status = wait_child(pid);
  EXPECT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << "child exited early, status=" << status;
  return last;
}

// Reopens after a crash and checks: state == model after all acked ops, or
// after one more op (the in-flight one may have reached the WAL). Advances
// model and next_op accordingly.
int g_rounds_with_journal = 0;  // kills that landed inside a checkpoint
uint64_t g_wal_records_redone = 0;

void check_recovered(const std::string& path, Model& model, uint64_t& next_op,
                     int64_t last_acked) {
  for (int64_t j = static_cast<int64_t>(next_op); j <= last_acked; ++j) apply_op(model, gen_op(j));
  next_op = static_cast<uint64_t>(last_acked + 1);
  auto db = DB::open(path, crashy_opts());
  ASSERT_NO_THROW(db->verify());
  if (db->stats().journal_pages_replayed > 0) ++g_rounds_with_journal;
  g_wal_records_redone += db->stats().recovered_records;
  const Model got = dump(*db);
  if (got == model) return;
  Model plus_one = model;
  apply_op(plus_one, gen_op(next_op));
  ASSERT_TRUE(got == plus_one) << "recovered state matches neither the acked prefix nor"
                                  " prefix+1 (got " << got.size() << " keys, expected "
                               << model.size() << ")";
  model = std::move(plus_one);
  ++next_op;
}

}  // namespace

TEST(Recovery, KillNineDuringWritesLosesNothingAcknowledged) {
  test::TempDir dir;
  const auto path = dir.file("crash.db");
  Model model;
  uint64_t next_op = 0;
  std::mt19937 rng(2024);
  constexpr int kRounds = 12;
  for (int round = 0; round < kRounds; ++round) {
    const int kill_after = 100 + static_cast<int>(rng() % 1500);
    const int64_t last = run_and_kill(path, next_op, kill_after, SyncMode::kNone);
    ASSERT_GE(last, static_cast<int64_t>(next_op) - 1);
    check_recovered(path, model, next_op, last);
    if (HasFatalFailure()) return;
  }
  EXPECT_GT(model.size(), 1000u);
  std::printf("[kill-9] %d rounds, %llu ops acknowledged, %llu WAL records redone, "
              "%d rounds recovered an interrupted checkpoint from the journal\n",
              kRounds,
              static_cast<unsigned long long>(next_op),
              static_cast<unsigned long long>(g_wal_records_redone), g_rounds_with_journal);
}

TEST(Recovery, KillNineWithFsyncCommits) {
  test::TempDir dir;
  const auto path = dir.file("crash.db");
  Model model;
  uint64_t next_op = 0;
  for (int round = 0; round < 3; ++round) {
    const int64_t last = run_and_kill(path, next_op, 300 + round * 200, SyncMode::kFsync);
    check_recovered(path, model, next_op, last);
    if (HasFatalFailure()) return;
  }
}

// SIGKILL at a random moment inside a large checkpoint (thousands of dirty
// pages), so some kills land after the journal is complete but before the
// in-place writes finish. Those must be repaired from the journal. Round 0
// is not killed; it measures how long the checkpoint takes so the kill
// delays of later rounds can be spread across it.
TEST(Recovery, KillNineDuringLargeCheckpoint) {
  constexpr uint64_t kExtra = 50000;
  std::mt19937 rng(99);
  int journal_rounds = 0;
  constexpr int kRounds = 10;
  int64_t lo_us = 0, hi_us = 1;  // kill window, from the calibration round
  for (int round = 0; round <= kRounds; ++round) {
    test::TempDir dir;
    const auto path = dir.file("ck.db");
    Options opts = crashy_opts();
    opts.pool_pages = 8192;
    opts.wal_checkpoint_bytes = 1ull << 40;
    int fds[2];
    ASSERT_EQ(::pipe(fds), 0);
    pid_t pid = ::fork();
    if (pid == 0) {
      ::close(fds[0]);
      auto db = DB::open(path, opts);
      for (uint64_t j = 0; j < 6000; ++j) db->write(gen_op(j));
      for (uint64_t i = 0; i < kExtra; ++i) db->put(key_of(100000 + i), value_of(i));
      int64_t msg = -1;
      if (::write(fds[1], &msg, sizeof msg) != sizeof msg) ::_exit(3);
      const auto t0 = std::chrono::steady_clock::now();
      db->checkpoint();
      msg = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
      if (::write(fds[1], &msg, sizeof msg) != sizeof msg) ::_exit(3);
      if (round == 0) {
        const auto st = db->stats();
        // Report phases so the parent can aim kills at the data-write phase.
        int64_t phases[2] = {static_cast<int64_t>(st.ckpt_journal_us),
                             static_cast<int64_t>(st.ckpt_data_us + st.ckpt_tail_us)};
        if (::write(fds[1], phases, sizeof phases) != sizeof phases) ::_exit(3);
      }
      ::pause();  // wait to be killed
      ::_exit(0);
    }
    ::close(fds[1]);
    int64_t msg = 0;
    ASSERT_EQ(::read(fds[0], &msg, sizeof msg), static_cast<ssize_t>(sizeof msg));
    if (round == 0) {
      ASSERT_EQ(::read(fds[0], &msg, sizeof msg), static_cast<ssize_t>(sizeof msg));
      int64_t phases[2];
      ASSERT_EQ(::read(fds[0], phases, sizeof phases), static_cast<ssize_t>(sizeof phases));
      // Kill anywhere from halfway through the journal write to the end of
      // the checkpoint, so both "journal incomplete" and "journal complete,
      // data partly written" crashes occur. Timing is noisy, which is fine.
      lo_us = phases[0] / 2;
      hi_us = std::max<int64_t>(phases[0] + phases[1], lo_us + 1);
      std::printf("[kill-9 in checkpoint] calibration: total %lld us, journal %lld us, "
                  "data+tail %lld us\n", static_cast<long long>(msg),
                  static_cast<long long>(phases[0]), static_cast<long long>(phases[1]));
    } else {
      ::usleep(static_cast<useconds_t>(lo_us + static_cast<int64_t>(rng() % (hi_us - lo_us))));
    }
    ::kill(pid, SIGKILL);
    ::close(fds[0]);
    wait_child(pid);
    Model model;
    for (uint64_t j = 0; j < 6000; ++j) apply_op(model, gen_op(j));
    for (uint64_t i = 0; i < kExtra; ++i) model[key_of(100000 + i)] = value_of(i);
    auto db = DB::open(path, opts);
    if (db->stats().journal_pages_replayed > 0) ++journal_rounds;
    ASSERT_NO_THROW(db->verify());
    ASSERT_TRUE(dump(*db) == model) << "round " << round;
  }
  std::printf("[kill-9 in checkpoint] %d killed rounds, %d repaired from a complete journal\n",
              kRounds, journal_rounds);
}

// Crash at each step of the checkpoint protocol, then again during recovery.
class CheckpointCrash : public ::testing::TestWithParam<const char*> {};

TEST_P(CheckpointCrash, RecoversAllCommittedData) {
  test::TempDir dir;
  const auto path = dir.file("fp.db");
  Options opts = crashy_opts();
  opts.pool_pages = 4096;  // no automatic checkpoints: we control them
  opts.wal_checkpoint_bytes = 1ull << 40;
  pid_t pid = ::fork();
  if (pid == 0) {
    try {
      auto db = DB::open(path, opts);
      for (uint64_t j = 0; j < 1500; ++j) db->write(gen_op(j));
      db->checkpoint();
      for (uint64_t j = 1500; j < 3000; ++j) db->write(gen_op(j));
      failpoint::set(GetParam());
      db->checkpoint();
    } catch (...) {
      ::_exit(4);
    }
    ::_exit(0);  // failpoint did not fire
  }
  int status = wait_child(pid);
  ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 77) << "status=" << status;

  // Crash a second time, at the same step, inside recovery's own checkpoint.
  pid = ::fork();
  if (pid == 0) {
    failpoint::set(GetParam());
    try {
      auto db = DB::open(path, opts);
    } catch (...) {
      ::_exit(4);
    }
    ::_exit(0);
  }
  status = wait_child(pid);
  ASSERT_TRUE(WIFEXITED(status)) << "status=" << status;
  const int code = WEXITSTATUS(status);
  ASSERT_TRUE(code == 77 || code == 0) << "code=" << code;

  Model model;
  for (uint64_t j = 0; j < 3000; ++j) apply_op(model, gen_op(j));
  auto db = DB::open(path, opts);
  ASSERT_NO_THROW(db->verify());
  EXPECT_TRUE(dump(*db) == model);
}

INSTANTIATE_TEST_SUITE_P(Steps, CheckpointCrash,
                         ::testing::Values("ckpt.before_journal_header", "ckpt.journal_synced",
                                           "ckpt.mid_data", "ckpt.data_synced",
                                           "ckpt.wal_reset"),
                         [](const auto& info) {
                           std::string s = info.param;
                           for (char& c : s) if (c == '.') c = '_';
                           return s;
                         });

TEST(Recovery, CrashWithoutCloseThenTornWalTail) {
  test::TempDir dir;
  const auto path = dir.file("torn.db");
  Options opts = crashy_opts();
  opts.pool_pages = 4096;
  opts.wal_checkpoint_bytes = 1ull << 40;
  pid_t pid = ::fork();
  if (pid == 0) {
    auto db = DB::open(path, opts);
    for (uint64_t j = 0; j < 2000; ++j) db->write(gen_op(j));
    ::_exit(0);  // no destructor, no checkpoint: everything is in the WAL
  }
  const int st = wait_child(pid);
  ASSERT_TRUE(WIFEXITED(st) && WEXITSTATUS(st) == 0);
  // Chop 7 bytes off the last record and append garbage, as a torn write would.
  const auto wal = path + "-wal";
  { Wal w(wal); w.replay([](uint64_t, std::string_view) {}); }  // trim zero tail
  const auto size = std::filesystem::file_size(wal);
  std::filesystem::resize_file(wal, size - 7);
  {
    int fd = ::open(wal.c_str(), O_WRONLY | O_APPEND);
    const char junk[] = "\x13\x37garbage-after-torn-record";
    ASSERT_GT(::write(fd, junk, sizeof junk), 0);
    ::close(fd);
  }
  Model model;
  for (uint64_t j = 0; j < 1999; ++j) apply_op(model, gen_op(j));  // last op torn away
  auto db = DB::open(path, opts);
  EXPECT_EQ(db->stats().recovered_records, 1999u);
  EXPECT_TRUE(dump(*db) == model);
}

TEST(Recovery, CorruptMagicIsRejected) {
  test::TempDir dir;
  const auto path = dir.file("bad.db");
  { auto db = DB::open(path, crashy_opts()); db->put("a", "b"); }
  int fd = ::open(path.c_str(), O_RDWR);
  const char z[8] = {};
  ASSERT_EQ(::pwrite(fd, z, 8, 0), 8);
  ::close(fd);
  EXPECT_THROW(DB::open(path, crashy_opts()), IoError);
}
