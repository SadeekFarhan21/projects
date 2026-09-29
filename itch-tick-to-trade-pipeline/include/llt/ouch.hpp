// OUCH-like order entry encoding written by the gateway to the simulated
// exchange connection. Loosely modeled on OUCH 4.2 Enter Order, trimmed to the
// fields this system uses, big-endian:
//
//   type 'O'(1) token(4) side(1) shares(4) stock(8) price(4) tif(1)  = 23 bytes
//
// tif 0 means immediate-or-cancel, the only kind the strategy sends.
#pragma once

#include <cstdint>
#include <cstring>

#include "llt/itch.hpp"
#include "llt/messages.hpp"

namespace llt::ouch {

inline constexpr std::size_t kLenEnter = 23;

struct EnterOrder {
    uint32_t token{0};
    char side{0};
    uint32_t shares{0};
    char stock[8]{};
    uint32_t price{0};
    uint8_t tif{0};

    bool operator==(const EnterOrder& o) const noexcept {
        return token == o.token && side == o.side && shares == o.shares &&
               std::memcmp(stock, o.stock, 8) == 0 && price == o.price && tif == o.tif;
    }
};

inline std::size_t encode_enter(uint8_t* p, const OrderRequest& o) noexcept {
    p[0] = 'O';
    itch::put32(p + 1, o.client_id);
    p[5] = static_cast<uint8_t>(o.side);
    itch::put32(p + 6, o.qty);
    std::memcpy(p + 10, o.stock, 8);
    itch::put32(p + 18, o.price);
    p[22] = 0;
    return kLenEnter;
}

inline bool decode_enter(const uint8_t* p, std::size_t n, EnterOrder& o) noexcept {
    if (n < kLenEnter || p[0] != 'O') return false;
    o.token = itch::be32(p + 1);
    o.side = static_cast<char>(p[5]);
    o.shares = itch::be32(p + 6);
    std::memcpy(o.stock, p + 10, 8);
    o.price = itch::be32(p + 18);
    o.tif = p[22];
    return true;
}

}  // namespace llt::ouch
