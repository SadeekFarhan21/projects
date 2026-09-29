// A market-data capture held entirely in memory before replay starts, so disk
// I/O never shows up in latency numbers.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llt {

struct Capture {
    std::vector<uint8_t> bytes;
    std::size_t message_count{0};  // framed messages, including unknown types
};

Capture capture_from_bytes(std::vector<uint8_t> bytes);
Capture load_capture(const std::string& path);  // throws std::runtime_error
void save_capture(const std::string& path, const std::vector<uint8_t>& bytes);

}  // namespace llt
