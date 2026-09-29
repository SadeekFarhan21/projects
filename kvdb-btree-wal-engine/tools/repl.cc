// Tiny interactive shell over kvdb.
//   kvdb_repl <db-path> [--sync=none|fsync|full]
#include <iostream>
#include <sstream>
#include <string>

#include "kvdb/db.h"

namespace {
void help() {
  std::cout << "commands:\n"
               "  put <key> <value>     insert or overwrite\n"
               "  get <key>\n"
               "  del <key>\n"
               "  scan [lo] [hi] [limit]  keys in [lo, hi), '-' = unbounded\n"
               "  stats                 engine counters and tree shape\n"
               "  checkpoint            flush dirty pages, empty the WAL\n"
               "  help | quit\n";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <db-path> [--sync=none|fsync|full]\n";
    return 2;
  }
  kvdb::Options opts;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--sync=none") opts.sync = kvdb::SyncMode::kNone;
    else if (a == "--sync=fsync") opts.sync = kvdb::SyncMode::kFsync;
    else if (a == "--sync=full") opts.sync = kvdb::SyncMode::kFullFsync;
  }
  std::unique_ptr<kvdb::DB> db;
  try {
    db = kvdb::DB::open(argv[1], opts);
  } catch (const std::exception& e) {
    std::cerr << "open failed: " << e.what() << "\n";
    return 1;
  }
  const auto st = db->stats();
  std::cout << "kvdb opened " << argv[1] << " (recovered " << st.recovered_records
            << " WAL records). type 'help'.\n";
  std::string line;
  while (std::cout << "kvdb> " << std::flush, std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    try {
      if (cmd.empty()) continue;
      if (cmd == "quit" || cmd == "exit") break;
      if (cmd == "help") {
        help();
      } else if (cmd == "put") {
        std::string k, v;
        in >> k;
        std::getline(in >> std::ws, v);
        db->put(k, v);
        std::cout << "OK\n";
      } else if (cmd == "get") {
        std::string k, v;
        in >> k;
        std::cout << (db->get(k, &v) ? v : "(not found)") << "\n";
      } else if (cmd == "del") {
        std::string k;
        in >> k;
        std::cout << (db->del(k) ? "deleted" : "(not found)") << "\n";
      } else if (cmd == "scan") {
        std::string lo = "-", hi = "-";
        long limit = 100;
        in >> lo >> hi >> limit;
        if (lo == "-") lo.clear();
        if (hi == "-") hi.clear();
        long n = 0;
        db->scan(lo, hi, [&](std::string_view k, std::string_view v) {
          std::cout << k << " = " << v << "\n";
          return ++n < limit;
        });
        std::cout << "(" << n << " rows)\n";
      } else if (cmd == "stats") {
        const auto s = db->stats();
        const auto t = db->verify();
        std::cout << "entries=" << t.entries << " height=" << t.height
                  << " leaves=" << t.leaf_pages << " internal=" << t.internal_pages
                  << " leaf_fill=" << t.leaf_fill << "\n"
                  << "commits=" << s.commits << " checkpoints=" << s.checkpoints
                  << " wal_bytes=" << s.wal_bytes << " pool_hits=" << s.pool.hits
                  << " pool_misses=" << s.pool.misses << "\n";
      } else if (cmd == "checkpoint") {
        db->checkpoint();
        std::cout << "OK\n";
      } else {
        std::cout << "unknown command, try 'help'\n";
      }
    } catch (const std::exception& e) {
      std::cout << "error: " << e.what() << "\n";
    }
  }
  return 0;
}
