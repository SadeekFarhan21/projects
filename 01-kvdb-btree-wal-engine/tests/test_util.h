#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace kvdb::test {

// Fresh directory under $TMPDIR, removed on destruction.
class TempDir {
 public:
  TempDir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/kvdb_test_XXXXXX";
    if (::mkdtemp(tmpl.data()) == nullptr) std::abort();
    path_ = tmpl;
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  std::string file(const std::string& name) const { return path_ + "/" + name; }

 private:
  std::string path_;
};

inline std::string key_of(uint64_t i) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "key%012llu", static_cast<unsigned long long>(i));
  return buf;
}

// Deterministic value whose length varies with i and a version number.
inline std::string value_of(uint64_t i, uint64_t version = 0) {
  std::string v = "v" + std::to_string(version) + ":" + std::to_string(i) + ":";
  v.resize(8 + (i * 7 + version * 13) % 120, static_cast<char>('a' + i % 26));
  return v;
}

}  // namespace kvdb::test
