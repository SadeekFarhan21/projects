#include "llt/capture.hpp"

#include <cstdio>
#include <stdexcept>

#include "llt/itch.hpp"

namespace llt {

Capture capture_from_bytes(std::vector<uint8_t> bytes) {
    Capture c;
    c.bytes = std::move(bytes);
    itch::FrameCursor cur{c.bytes.data(), c.bytes.data() + c.bytes.size()};
    const uint8_t* msg;
    uint16_t len;
    while (cur.next(msg, len)) ++c.message_count;
    return c;
}

Capture load_capture(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open capture: " + path);
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<std::size_t>(sz));
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (got != buf.size()) throw std::runtime_error("short read: " + path);
    return capture_from_bytes(std::move(buf));
}

void save_capture(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write: " + path);
    const std::size_t put = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (put != bytes.size()) throw std::runtime_error("short write: " + path);
}

}  // namespace llt
