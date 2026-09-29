// Normalized in-process message types that flow between pipeline stages.
// All are trivially copyable fixed-size structs so they can live in ring slots.
#pragma once

#include <cstdint>
#include <type_traits>

namespace llt {

// One decoded market-data message. The feed handler fills only the fields the
// message type carries; E/X/D do not carry price or side (the book looks them
// up by order_ref, exactly like real ITCH).
struct MdEvent {
    uint64_t order_ref{0};
    uint64_t exch_ts{0};  // ns since midnight, from the ITCH header
    uint64_t t_in{0};     // local counter ticks when the feed handler picked it up
    uint32_t price{0};    // ITCH Price(4): 4 implied decimals
    uint32_t qty{0};      // shares added / executed / cancelled
    uint16_t locate{0};
    char type{0};  // 'A','E','X','D','R', or 'Z' (end-of-stream sentinel)
    char side{0};  // 'B' or 'S' for 'A'
    char stock[8]{};
};
static_assert(sizeof(MdEvent) == 48);
static_assert(std::is_trivially_copyable_v<MdEvent>);

// What the strategy asks the gateway to send.
struct OrderRequest {
    uint64_t t_in{0};      // copied from the triggering MdEvent
    uint64_t t_decide{0};  // after risk accepted it
    uint32_t client_id{0};
    uint32_t price{0};
    uint32_t qty{0};
    uint16_t locate{0};
    char side{0};  // 'B' / 'S', or 'Z' sentinel
    char stock[8]{};
};
static_assert(sizeof(OrderRequest) == 40);
static_assert(std::is_trivially_copyable_v<OrderRequest>);

// Bytes on the simulated exchange connection. len == 0 is the shutdown sentinel.
struct WireFrame {
    uint16_t len{0};
    uint8_t data[46]{};
};
static_assert(sizeof(WireFrame) == 48);

// Per-order latency record, indexed by client_id.
struct OrderTiming {
    uint64_t t_in{0};
    uint64_t t_decide{0};
    uint64_t t_out{0};  // frame encoded, about to be published to the connection
    uint64_t t_rx{0};   // simulated exchange read it
};

}  // namespace llt
