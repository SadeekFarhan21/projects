#pragma once
// Tiny helpers shared by the command-line tools.
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace cli {

struct Args {
  std::vector<std::string> positional;
  std::vector<std::pair<std::string, std::string>> opts;  // --key value or --flag ""

  bool has(std::string_view k) const {
    for (auto& [a, b] : opts)
      if (a == k) return true;
    return false;
  }
  std::string get(std::string_view k, std::string def = "") const {
    for (auto& [a, b] : opts)
      if (a == k) return b;
    return def;
  }
  double num(std::string_view k, double def) const {
    auto v = get(k);
    return v.empty() ? def : std::strtod(v.c_str(), nullptr);
  }
};

// Options listed in `flags` take no value; every other --opt takes one.
inline Args parse(int argc, char** argv, std::vector<std::string_view> flags = {}) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      std::string key = s.substr(2);
      bool is_flag = false;
      for (auto f : flags)
        if (f == key) is_flag = true;
      if (is_flag || i + 1 >= argc) a.opts.emplace_back(key, "");
      else a.opts.emplace_back(key, argv[++i]);
    } else {
      a.positional.push_back(s);
    }
  }
  return a;
}

inline double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// CPU time consumed by the calling thread / the whole process. The dev machine
// was shared with other heavy jobs while this was written, so wall-clock
// numbers are noisy; CPU time is reported next to them because it is not
// inflated by time spent waiting for a core.
inline double thread_cpu_s() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_nsec) * 1e-9;
}
inline double process_cpu_s() {
  timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_nsec) * 1e-9;
}

// 1-minute load average, recorded next to every timing result.
inline double load_avg_1m() {
  double l[3] = {0, 0, 0};
  return getloadavg(l, 3) > 0 ? l[0] : -1.0;
}

[[noreturn]] inline void die(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

inline std::string json_escape(std::string_view s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      o += buf;
    } else o += c;
  }
  return o;
}

}  // namespace cli
