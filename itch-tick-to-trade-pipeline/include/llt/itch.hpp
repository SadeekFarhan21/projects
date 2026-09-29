// ITCH-like binary market data format.
//
// Framing: each message is preceded by a 2-byte big-endian length, the same
// framing NASDAQ uses in its ITCH 5.0 binary files and SoupBinTCP.
//
// Message bodies ('A','E','X','D') use the exact ITCH 5.0 field layouts and
// big-endian encoding:
//
//   common header  type(1) stock_locate(2) tracking_number(2) timestamp(6, ns)
//   'A' Add Order        + order_ref(8) side(1) shares(4) stock(8) price(4)  = 36
//   'E' Order Executed   + order_ref(8) executed_shares(4) match_number(8)   = 31
//   'X' Order Cancel     + order_ref(8) cancelled_shares(4)                  = 23
//   'D' Order Delete     + order_ref(8)                                      = 19
//
// 'R' is a reduced Stock Directory (header + stock(8) = 19 bytes). Real ITCH
// 'R' is 39 bytes with fields we do not use. Any other type byte is skipped.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "llt/messages.hpp"

namespace llt::itch {

inline constexpr std::size_t kLenAdd = 36;
inline constexpr std::size_t kLenExec = 31;
inline constexpr std::size_t kLenCancel = 23;
inline constexpr std::size_t kLenDelete = 19;
inline constexpr std::size_t kLenDir = 19;
inline constexpr std::size_t kHeaderLen = 11;

// ---- big-endian helpers (memcpy + bswap compiles to ldr + rev) ----
inline uint16_t be16(const uint8_t* p) noexcept {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return __builtin_bswap16(v);
}
inline uint32_t be32(const uint8_t* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return __builtin_bswap32(v);
}
inline uint64_t be64(const uint8_t* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return __builtin_bswap64(v);
}
inline uint64_t be48(const uint8_t* p) noexcept {
    return (static_cast<uint64_t>(be16(p)) << 32) | be32(p + 2);
}
inline void put16(uint8_t* p, uint16_t v) noexcept {
    v = __builtin_bswap16(v);
    std::memcpy(p, &v, 2);
}
inline void put32(uint8_t* p, uint32_t v) noexcept {
    v = __builtin_bswap32(v);
    std::memcpy(p, &v, 4);
}
inline void put64(uint8_t* p, uint64_t v) noexcept {
    v = __builtin_bswap64(v);
    std::memcpy(p, &v, 8);
}
inline void put48(uint8_t* p, uint64_t v) noexcept {
    put16(p, static_cast<uint16_t>(v >> 32));
    put32(p + 2, static_cast<uint32_t>(v));
}

// Decode one message body (no length prefix). Returns false for unknown types
// or truncated bodies; the caller skips those. Zero-copy, no allocation.
inline bool decode(const uint8_t* p, std::size_t n, MdEvent& e) noexcept {
    if (n < kHeaderLen) return false;
    const char t = static_cast<char>(p[0]);
    e.type = t;
    e.locate = be16(p + 1);
    e.exch_ts = be48(p + 5);
    switch (t) {
        case 'A':
            if (n < kLenAdd) return false;
            e.order_ref = be64(p + 11);
            e.side = static_cast<char>(p[19]);
            e.qty = be32(p + 20);
            std::memcpy(e.stock, p + 24, 8);
            e.price = be32(p + 32);
            return true;
        case 'E':
            if (n < kLenExec) return false;
            e.order_ref = be64(p + 11);
            e.qty = be32(p + 19);
            e.price = 0;
            e.side = 0;
            return true;
        case 'X':
            if (n < kLenCancel) return false;
            e.order_ref = be64(p + 11);
            e.qty = be32(p + 19);
            e.price = 0;
            e.side = 0;
            return true;
        case 'D':
            if (n < kLenDelete) return false;
            e.order_ref = be64(p + 11);
            e.qty = 0;
            e.price = 0;
            e.side = 0;
            return true;
        case 'R':
            if (n < kLenDir) return false;
            std::memcpy(e.stock, p + 11, 8);
            e.order_ref = 0;
            e.qty = 0;
            e.price = 0;
            e.side = 0;
            return true;
        default:
            return false;
    }
}

// Walks a buffer of length-prefixed frames.
struct FrameCursor {
    const uint8_t* p;
    const uint8_t* end;

    bool next(const uint8_t*& msg, uint16_t& len) noexcept {
        if (end - p < 2) return false;
        len = be16(p);
        if (end - p - 2 < static_cast<std::ptrdiff_t>(len)) return false;  // truncated tail
        msg = p + 2;
        p += 2 + len;
        return true;
    }
};

// Appends framed messages to a byte vector. Used by the generator and tests,
// never on the hot path.
class Writer {
public:
    explicit Writer(std::vector<uint8_t>& out) : out_(out) {}

    void directory(uint16_t locate, uint64_t ts, const char stock[8]) {
        uint8_t* p = begin(kLenDir, 'R', locate, ts);
        std::memcpy(p + 11, stock, 8);
    }
    void add(uint16_t locate, uint64_t ts, uint64_t ref, char side, uint32_t shares,
             const char stock[8], uint32_t price) {
        uint8_t* p = begin(kLenAdd, 'A', locate, ts);
        put64(p + 11, ref);
        p[19] = static_cast<uint8_t>(side);
        put32(p + 20, shares);
        std::memcpy(p + 24, stock, 8);
        put32(p + 32, price);
    }
    void executed(uint16_t locate, uint64_t ts, uint64_t ref, uint32_t shares, uint64_t match) {
        uint8_t* p = begin(kLenExec, 'E', locate, ts);
        put64(p + 11, ref);
        put32(p + 19, shares);
        put64(p + 23, match);
    }
    void cancel(uint16_t locate, uint64_t ts, uint64_t ref, uint32_t shares) {
        uint8_t* p = begin(kLenCancel, 'X', locate, ts);
        put64(p + 11, ref);
        put32(p + 19, shares);
    }
    void del(uint16_t locate, uint64_t ts, uint64_t ref) {
        uint8_t* p = begin(kLenDelete, 'D', locate, ts);
        put64(p + 11, ref);
    }

private:
    uint8_t* begin(std::size_t len, char type, uint16_t locate, uint64_t ts) {
        const std::size_t at = out_.size();
        out_.resize(at + 2 + len);
        uint8_t* f = out_.data() + at;
        put16(f, static_cast<uint16_t>(len));
        uint8_t* p = f + 2;
        p[0] = static_cast<uint8_t>(type);
        put16(p + 1, locate);
        put16(p + 3, 0);  // tracking number
        put48(p + 5, ts);
        return p;
    }
    std::vector<uint8_t>& out_;
};

}  // namespace llt::itch
