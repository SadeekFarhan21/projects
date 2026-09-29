// Throughput and latency benchmark: kvdb vs std::map vs SQLite.
//
//   kvdb_bench --exp=main --n=1000000 --out=results/bench_main.csv
//   kvdb_bench --exp=sync --n=3000    --out=results/bench_sync.csv
//   kvdb_bench --exp=walgrow --n=3000 --out=results/bench_walgrow.csv
//
// Reports wall-clock ops/s and ops per CPU-second (see cpu_seconds()).
// Every operation is timed individually with steady_clock (about 20-40 ns of
// overhead per op, included in the numbers). Latencies are exact percentiles
// over all ops, not sampled.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include "kvdb/db.h"

#ifdef KVDB_HAVE_SQLITE
#include <sqlite3.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

// User + system CPU seconds of this process. The benchmark machine is shared
// and heavily oversubscribed, so wall-clock throughput includes time spent
// waiting for a core; CPU time does not.
double cpu_seconds() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  auto tv = [](timeval t) { return t.tv_sec + t.tv_usec / 1e6; };
  return tv(ru.ru_utime) + tv(ru.ru_stime);
}

struct Args {
  std::string exp = "main";
  uint64_t n = 1000000;
  size_t value_size = 100;
  std::string out = "results/bench.csv";
  std::string dir;  // scratch dir for database files
};

std::string key_of(uint64_t i) {
  char buf[24];
  std::snprintf(buf, sizeof buf, "user%012llu", static_cast<unsigned long long>(i));
  return buf;  // 16 bytes; lexicographic order == numeric order
}

std::string value_of(uint64_t i, size_t size) {
  std::string v(size, 'v');
  // Mix the index in so values are not all identical.
  for (size_t j = 0; j < std::min<size_t>(size, 8); ++j) v[j] = static_cast<char>('a' + (i >> (4 * j)) % 26);
  return v;
}

struct Result {
  std::string exp, engine, workload;
  uint64_t n = 0;
  uint64_t batch = 1;
  double seconds = 0;
  double cpu_seconds = 0;
  double p50 = 0, p99 = 0, p999 = 0, max = 0;  // ns
  std::string note;
  double load1 = 0;  // 1-minute load average when the measurement ended
  double ops_per_sec() const { return n / seconds; }
};

std::vector<Result> g_results;

// Runs op(i) for each i in order, timing each call.
Result timed(const std::string& exp, const std::string& engine, const std::string& workload,
             const std::vector<uint64_t>& order, const std::function<void(uint64_t)>& op,
             uint64_t ops_per_call = 1) {
  std::vector<uint32_t> lat;
  lat.reserve(order.size());
  const double cpu0 = cpu_seconds();
  const auto start = Clock::now();
  for (uint64_t i : order) {
    const auto t0 = Clock::now();
    op(i);
    const auto t1 = Clock::now();
    lat.push_back(static_cast<uint32_t>(
        std::min<int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                          UINT32_MAX)));
  }
  const double secs = std::chrono::duration<double>(Clock::now() - start).count();
  const double cpu = cpu_seconds() - cpu0;
  std::sort(lat.begin(), lat.end());
  auto pct = [&](double p) {
    return static_cast<double>(lat[std::min(lat.size() - 1, static_cast<size_t>(p * lat.size()))]);
  };
  Result r{exp, engine, workload, order.size() * ops_per_call, ops_per_call, secs, cpu,
           pct(0.50), pct(0.99), pct(0.999), static_cast<double>(lat.back()), "", 0};
  double la[3] = {0, 0, 0};
  getloadavg(la, 3);
  r.load1 = la[0];
  std::printf("%-6s %-16s %-12s n=%-8llu batch=%-4llu %10.0f ops/s %10.0f ops/cpu-s  p50=%7.0f ns  p99=%8.0f ns  p99.9=%9.0f ns  load=%.1f\n",
              exp.c_str(), engine.c_str(), workload.c_str(), static_cast<unsigned long long>(r.n),
              static_cast<unsigned long long>(ops_per_call), r.ops_per_sec(), r.n / std::max(cpu, 1e-9), r.p50, r.p99, r.p999, r.load1);
  std::fflush(stdout);
  g_results.push_back(r);
  return r;
}

void remove_db(const std::string& p) {
  for (const char* suf : {"", "-wal", "-journal", "-shm"}) std::filesystem::remove(p + suf);
}

uint64_t db_bytes(const std::string& p) {
  uint64_t total = 0;
  for (const char* suf : {"", "-wal", "-journal"}) {
    std::error_code ec;
    auto s = std::filesystem::file_size(p + suf, ec);
    if (!ec) total += s;
  }
  return total;
}

// ---------------------------------------------------------------- engines ---

struct Engine {
  virtual ~Engine() = default;
  virtual void put(const std::string& k, const std::string& v) = 0;
  virtual bool get(const std::string& k, std::string* v) = 0;
  virtual uint64_t scan_all() = 0;
  virtual std::string note() { return ""; }
};

struct KvdbEngine : Engine {
  std::unique_ptr<kvdb::DB> db;
  std::string path;
  KvdbEngine(const std::string& p, kvdb::Options o) : path(p) {
    remove_db(p);
    db = kvdb::DB::open(p, o);
  }
  void put(const std::string& k, const std::string& v) override { db->put(k, v); }
  bool get(const std::string& k, std::string* v) override { return db->get(k, v); }
  uint64_t scan_all() override {
    uint64_t n = 0;
    db->scan("", "", [&](std::string_view, std::string_view) { return ++n, true; });
    return n;
  }
  std::string note() override {
    const auto t = db->verify();
    const auto s = db->stats();
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "height=%u leaves=%llu fill=%.2f checkpoints=%llu pool_hit=%.3f file_mb=%.1f",
                  t.height, static_cast<unsigned long long>(t.leaf_pages), t.leaf_fill,
                  static_cast<unsigned long long>(s.checkpoints),
                  double(s.pool.hits) / double(std::max<uint64_t>(1, s.pool.hits + s.pool.misses)),
                  db_bytes(path) / 1048576.0);
    return buf;
  }
};

struct MapEngine : Engine {
  std::map<std::string, std::string> m;
  void put(const std::string& k, const std::string& v) override { m[k] = v; }
  bool get(const std::string& k, std::string* v) override {
    auto it = m.find(k);
    if (it == m.end()) return false;
    *v = it->second;
    return true;
  }
  uint64_t scan_all() override {
    uint64_t n = 0;
    for (auto& kv : m) n += kv.first.size() > 0;
    return n;
  }
};

#ifdef KVDB_HAVE_SQLITE
struct SqliteEngine : Engine {
  sqlite3* db = nullptr;
  sqlite3_stmt *put_s = nullptr, *get_s = nullptr;
  std::string path;
  void exec(const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
      std::fprintf(stderr, "sqlite: %s: %s\n", sql, err);
      std::exit(1);
    }
  }
  // synchronous: "OFF" | "NORMAL" | "FULL"
  SqliteEngine(const std::string& p, const char* synchronous, bool fullfsync) : path(p) {
    remove_db(p);
    sqlite3_open(p.c_str(), &db);
    exec("PRAGMA journal_mode=WAL");
    exec((std::string("PRAGMA synchronous=") + synchronous).c_str());
    exec(fullfsync ? "PRAGMA fullfsync=ON" : "PRAGMA fullfsync=OFF");
    exec("PRAGMA cache_size=-262144");  // 256 MiB, same as the kvdb pool
    exec("CREATE TABLE kv(k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID");
    sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO kv(k, v) VALUES(?, ?)", -1, &put_s, nullptr);
    sqlite3_prepare_v2(db, "SELECT v FROM kv WHERE k = ?", -1, &get_s, nullptr);
  }
  ~SqliteEngine() override {
    sqlite3_finalize(put_s);
    sqlite3_finalize(get_s);
    sqlite3_close(db);
  }
  void put(const std::string& k, const std::string& v) override {
    sqlite3_bind_blob(put_s, 1, k.data(), static_cast<int>(k.size()), SQLITE_STATIC);
    sqlite3_bind_blob(put_s, 2, v.data(), static_cast<int>(v.size()), SQLITE_STATIC);
    if (sqlite3_step(put_s) != SQLITE_DONE) std::abort();
    sqlite3_reset(put_s);
  }
  bool get(const std::string& k, std::string* v) override {
    sqlite3_bind_blob(get_s, 1, k.data(), static_cast<int>(k.size()), SQLITE_STATIC);
    bool found = false;
    if (sqlite3_step(get_s) == SQLITE_ROW) {
      v->assign(static_cast<const char*>(sqlite3_column_blob(get_s, 0)),
                static_cast<size_t>(sqlite3_column_bytes(get_s, 0)));
      found = true;
    }
    sqlite3_reset(get_s);
    return found;
  }
  uint64_t scan_all() override {
    sqlite3_stmt* s;
    sqlite3_prepare_v2(db, "SELECT k, v FROM kv ORDER BY k", -1, &s, nullptr);
    uint64_t n = 0;
    while (sqlite3_step(s) == SQLITE_ROW) n += sqlite3_column_bytes(s, 0) > 0;
    sqlite3_finalize(s);
    return n;
  }
  std::string note() override {
    char buf[64];
    std::snprintf(buf, sizeof buf, "file_mb=%.1f", db_bytes(path) / 1048576.0);
    return buf;
  }
};
#endif

// ------------------------------------------------------------ experiments ---

// Fill in one order, then point-read in sequential and random order, then
// scan. Done twice per engine: sequential fill and random fill.
void run_main(const Args& a) {
  std::vector<uint64_t> seq(a.n);
  std::iota(seq.begin(), seq.end(), 0);
  std::vector<uint64_t> rnd = seq;
  std::shuffle(rnd.begin(), rnd.end(), std::mt19937_64(12345));
  std::vector<uint64_t> rnd_read = seq;
  std::shuffle(rnd_read.begin(), rnd_read.end(), std::mt19937_64(999));
  std::vector<std::string> keys(a.n);
  for (uint64_t i = 0; i < a.n; ++i) keys[i] = key_of(i);
  const std::string val = value_of(7, a.value_size);

  struct Spec {
    std::string name;
    std::function<std::unique_ptr<Engine>()> make;
  };
  std::vector<Spec> specs;
  specs.push_back({"std::map", [] { return std::make_unique<MapEngine>(); }});
  auto kv_opts = [](size_t pool_pages) {
    kvdb::Options o;
    o.sync = kvdb::SyncMode::kNone;
    o.pool_pages = pool_pages;
    return o;
  };
  specs.push_back({"kvdb", [&] { return std::make_unique<KvdbEngine>(a.dir + "/kv.db", kv_opts(65536)); }});
  specs.push_back({"kvdb_pool16MiB", [&] { return std::make_unique<KvdbEngine>(a.dir + "/kv.db", kv_opts(4096)); }});
#ifdef KVDB_HAVE_SQLITE
  specs.push_back({"sqlite", [&] { return std::make_unique<SqliteEngine>(a.dir + "/sq.db", "OFF", false); }});
#endif

  for (const auto& spec : specs) {
    for (const char* fill : {"seq", "random"}) {
      auto e = spec.make();
      const auto& order = std::string(fill) == "seq" ? seq : rnd;
      std::string out;
      timed("main", spec.name, std::string("fill") + fill, order,
            [&](uint64_t i) { e->put(keys[i], val); });
      g_results.back().note = e->note();
      uint64_t misses = 0;
      timed("main", spec.name, std::string("readseq@") + fill, seq,
            [&](uint64_t i) { misses += !e->get(keys[i], &out); });
      timed("main", spec.name, std::string("readrandom@") + fill, rnd_read,
            [&](uint64_t i) { misses += !e->get(keys[i], &out); });
      uint64_t scanned = 0;
      timed("main", spec.name, std::string("scanall@") + fill, std::vector<uint64_t>{0},
            [&](uint64_t) { scanned = e->scan_all(); }, a.n);
      if (misses != 0 || scanned != a.n) {
        std::fprintf(stderr, "CORRECTNESS FAILURE %s: misses=%llu scanned=%llu\n",
                     spec.name.c_str(), static_cast<unsigned long long>(misses),
                     static_cast<unsigned long long>(scanned));
        std::exit(1);
      }
    }
  }
}

// Commit latency under each durability mode, and group commit via batches.
void run_sync(const Args& a) {
  const std::string val = value_of(7, a.value_size);
  std::vector<uint64_t> order(a.n);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937_64(5));
  const std::string p = a.dir + "/sync.db";

  const std::pair<const char*, kvdb::SyncMode> modes[] = {
      {"none", kvdb::SyncMode::kNone},
      {"fsync", kvdb::SyncMode::kFsync},
      {"fullfsync", kvdb::SyncMode::kFullFsync}};
  for (const auto& [name, mode] : modes) {
    kvdb::Options o;
    o.sync = mode;
    KvdbEngine e(p, o);
    timed("sync", "kvdb", std::string("put_") + name, order,
          [&](uint64_t i) { e.put(key_of(i), val); });
  }
#ifdef KVDB_HAVE_SQLITE
  const std::tuple<const char*, const char*, bool> sq[] = {
      {"none", "OFF", false}, {"fsync", "FULL", false}, {"fullfsync", "FULL", true}};
  for (const auto& [name, syn, full] : sq) {
    SqliteEngine e(a.dir + "/sync_sq.db", syn, full);
    timed("sync", "sqlite", std::string("put_") + name, order,
          [&](uint64_t i) { e.put(key_of(i), val); });
  }
#endif
  // Group commit: one F_FULLFSYNC per batch of b puts.
  for (uint64_t b : {1, 4, 16, 64, 256}) {
    kvdb::Options o;
    o.sync = kvdb::SyncMode::kFullFsync;
    KvdbEngine e(p, o);
    const uint64_t batches = std::max<uint64_t>(1, a.n / b);
    std::vector<uint64_t> idx(batches);
    std::iota(idx.begin(), idx.end(), 0);
    kvdb::WriteBatch wb;
    timed("sync", "kvdb", "batch_fullfsync", idx, [&](uint64_t j) {
      wb.clear();
      for (uint64_t i = j * b; i < (j + 1) * b; ++i) wb.put(key_of(order[i % a.n]), val);
      e.db->write(wb);
    }, b);
  }
}

// Why the WAL preallocates: cost of a 128-byte log append that extends the
// file versus one that lands inside an already-sized (zero-filled) file,
// each followed by the given sync. Uses raw pwrite so only the file system
// is measured.
void run_walgrow(const Args& a) {
  const std::string p = a.dir + "/walgrow.log";
  const std::string rec(128, 'r');
  std::vector<uint64_t> order(a.n);
  std::iota(order.begin(), order.end(), 0);
  const std::pair<const char*, kvdb::SyncMode> modes[] = {
      {"none", kvdb::SyncMode::kNone}, {"fsync", kvdb::SyncMode::kFsync},
      {"fullfsync", kvdb::SyncMode::kFullFsync}};
  for (const auto& [name, mode] : modes) {
    for (bool prealloc : {false, true}) {
      std::filesystem::remove(p);
      const int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
      if (fd < 0) std::exit(1);
      if (prealloc && ::ftruncate(fd, static_cast<off_t>(a.n * rec.size())) != 0) std::exit(1);
      kvdb::sync_fd(fd, kvdb::SyncMode::kFullFsync);
      timed("walgrow", prealloc ? "prealloc" : "extend", std::string("append_") + name, order,
            [&](uint64_t i) {
              if (::pwrite(fd, rec.data(), rec.size(), static_cast<off_t>(i * rec.size())) !=
                  static_cast<ssize_t>(rec.size()))
                std::exit(1);
              kvdb::sync_fd(fd, mode);
            });
      ::close(fd);
    }
  }
  std::filesystem::remove(p);
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto val = [&](const char* flag) -> const char* {
      const size_t n = std::strlen(flag);
      return s.compare(0, n, flag) == 0 ? s.c_str() + n : nullptr;
    };
    if (auto v = val("--exp=")) a.exp = v;
    else if (auto v = val("--n=")) a.n = std::strtoull(v, nullptr, 10);
    else if (auto v = val("--value=")) a.value_size = std::strtoull(v, nullptr, 10);
    else if (auto v = val("--out=")) a.out = v;
    else if (auto v = val("--dir=")) a.dir = v;
    else {
      std::fprintf(stderr, "unknown flag %s\n", argv[i]);
      return 2;
    }
  }
  if (a.dir.empty()) {
    const char* t = std::getenv("TMPDIR");
    a.dir = std::string(t ? t : "/tmp") + "/kvdb_bench";
  }
  std::filesystem::create_directories(a.dir);
  double load[3] = {0, 0, 0};
  getloadavg(load, 3);
  std::printf("# exp=%s n=%llu key=16B value=%zuB dir=%s loadavg=%.1f %.1f %.1f\n", a.exp.c_str(),
              static_cast<unsigned long long>(a.n), a.value_size, a.dir.c_str(), load[0], load[1],
              load[2]);
#ifdef KVDB_HAVE_SQLITE
  std::printf("# sqlite %s\n", sqlite3_libversion());
#endif
  if (a.exp == "main") run_main(a);
  else if (a.exp == "sync") run_sync(a);
  else if (a.exp == "walgrow") run_walgrow(a);
  else {
    std::fprintf(stderr, "unknown --exp\n");
    return 2;
  }
  std::ofstream out(a.out);
  out << "exp,engine,workload,n,batch,seconds,ops_per_sec,cpu_seconds,ops_per_cpu_sec,p50_ns,p99_ns,p999_ns,max_ns,load1,note\n";
  for (const auto& r : g_results) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s,%s,%s,%llu,%llu,%.4f,%.0f,%.4f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,\"%s\"\n",
                  r.exp.c_str(), r.engine.c_str(), r.workload.c_str(),
                  static_cast<unsigned long long>(r.n), static_cast<unsigned long long>(r.batch),
                  r.seconds, r.ops_per_sec(), r.cpu_seconds,
                  r.n / std::max(r.cpu_seconds, 1e-9), r.p50, r.p99, r.p999, r.max, r.load1, r.note.c_str());
    out << buf;
  }
  std::filesystem::remove_all(a.dir);
  return 0;
}
