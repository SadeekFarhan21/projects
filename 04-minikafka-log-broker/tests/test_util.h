#pragma once

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <random>
#include <string>

namespace mk::test {

// A fresh directory under the system temp dir, removed on destruction.
class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    std::random_device rd;
    path_ = std::filesystem::temp_directory_path() /
            ("minikafka-test-" + std::to_string(::getpid()) + "-" + std::to_string(counter++) + "-" +
             std::to_string(rd()));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace mk::test
