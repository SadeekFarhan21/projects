#pragma once
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "se/index.hpp"
#include "se/index_builder.hpp"

namespace setest {

// Unique temp path per call; removed when the object dies.
struct TempPath {
  std::string path;
  TempPath() {
    static std::atomic<int> counter{0};
    path = (std::filesystem::temp_directory_path() /
            ("se_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ".idx"))
               .string();
  }
  TempPath(const TempPath&) = delete;  // a copy would delete the file twice
  TempPath& operator=(const TempPath&) = delete;
  ~TempPath() { std::filesystem::remove(path); }
};

struct Doc {
  std::string id, title, text;
};

struct Built {
  std::unique_ptr<TempPath> tmp;
  se::BuildStats stats;
  se::Index index;
};

inline std::unique_ptr<Built> build(const std::vector<Doc>& docs, se::TokenizerOptions opts) {
  se::IndexBuilder b(opts);
  for (const auto& d : docs) b.add_document(d.id, d.title, d.text);
  auto tmp = std::make_unique<TempPath>();
  auto st = b.write(tmp->path);
  auto ix = se::Index::open(tmp->path);
  return std::unique_ptr<Built>(new Built{std::move(tmp), st, std::move(ix)});
}

inline se::TokenizerOptions plain() {
  se::TokenizerOptions o;
  o.stem = false;
  o.stopwords = false;
  return o;
}

}  // namespace setest
