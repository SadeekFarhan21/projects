// minute_snap: stream filter for tardis.dev Deribit options_chain CSV.
//
// Reads CSV on stdin, keeps rows whose symbol starts with one of the given
// prefixes (default "BTC-" and "ETH-"), and emits, per symbol, the last row
// observed strictly before each one minute boundary. Each emitted line is
// prefixed with snap_ts (microseconds, the boundary the row is valid at).
// Downstream code forward fills the per-symbol series onto a full grid.
//
// Why C++: the raw file is ~100 GB uncompressed per day. awk on macOS is
// too slow to keep up with the download, and Python line loops are slower.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct Slot {
  int64_t minute = -1;
  int64_t ts = -1;
  std::string line;
};

static constexpr int64_t kMinuteUs = 60'000'000;

int main(int argc, char** argv) {
  std::vector<std::string> prefixes;
  for (int i = 1; i < argc; ++i) prefixes.emplace_back(argv[i]);
  if (prefixes.empty()) prefixes = {"BTC-", "ETH-"};

  std::unordered_map<std::string, Slot> slots;
  slots.reserve(8192);
  char* buf = nullptr;
  size_t cap = 0;
  ssize_t n;
  bool header = true;
  uint64_t rows_in = 0, rows_kept = 0, rows_out = 0;

  auto emit = [&](const Slot& s) {
    std::fprintf(stdout, "%lld,%s\n", (long long)((s.minute + 1) * kMinuteUs), s.line.c_str());
    ++rows_out;
  };

  while ((n = getline(&buf, &cap, stdin)) > 0) {
    if (buf[n - 1] == '\n') buf[--n] = '\0';
    if (n > 0 && buf[n - 1] == '\r') buf[--n] = '\0';
    if (header) {
      std::fprintf(stdout, "snap_ts,%s\n", buf);
      header = false;
      continue;
    }
    ++rows_in;
    // fields: exchange,symbol,timestamp,...
    const char* c1 = std::strchr(buf, ',');
    if (!c1) continue;
    const char* sym = c1 + 1;
    const char* c2 = std::strchr(sym, ',');
    if (!c2) continue;
    std::string_view symbol(sym, c2 - sym);
    bool keep = false;
    for (const auto& p : prefixes)
      if (symbol.substr(0, p.size()) == p) { keep = true; break; }
    if (!keep) continue;
    ++rows_kept;
    int64_t ts = std::strtoll(c2 + 1, nullptr, 10);
    int64_t minute = ts / kMinuteUs;
    auto it = slots.find(std::string(symbol));
    if (it == slots.end()) {
      Slot s{minute, ts, std::string(buf, n)};
      slots.emplace(std::string(symbol), std::move(s));
      continue;
    }
    Slot& s = it->second;
    if (minute > s.minute) {
      emit(s);
      s.minute = minute;
      s.ts = ts;
      s.line.assign(buf, n);
    } else if (ts >= s.ts) {  // same minute (or late row): keep the newest
      s.ts = ts;
      s.line.assign(buf, n);
    }
  }
  for (auto& [k, s] : slots) emit(s);
  std::fprintf(stderr, "minute_snap rows_in=%llu kept=%llu out=%llu symbols=%zu\n",
               (unsigned long long)rows_in, (unsigned long long)rows_kept,
               (unsigned long long)rows_out, slots.size());
  free(buf);
  return 0;
}
