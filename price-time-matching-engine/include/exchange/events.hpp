// Event definitions. The engine is a pure function from an ordered stream of
// input events to an ordered stream of output events. Both streams are plain
// data so they can be logged, replayed and diffed.
#pragma once

#include "exchange/types.hpp"

#include <cstdint>
#include <string>

namespace exch {

// ---------------------------------------------------------------- inputs

enum class InputKind : std::uint8_t { New = 0, Cancel, Modify };

struct InputEvent {
    InputKind kind{InputKind::New};
    OrderId id{0};
    Side side{Side::Buy};              // New only
    OrderType type{OrderType::Limit};  // New only
    Price price{0};                    // New (ignored for Market), Modify
    Qty qty{0};                        // New, Modify (new open quantity)

    static InputEvent new_order(OrderId id, Side s, OrderType t, Price p, Qty q) {
        return InputEvent{InputKind::New, id, s, t, p, q};
    }
    static InputEvent cancel(OrderId id) {
        InputEvent e;
        e.kind = InputKind::Cancel;
        e.id = id;
        return e;
    }
    static InputEvent modify(OrderId id, Price p, Qty q) {
        InputEvent e;
        e.kind = InputKind::Modify;
        e.id = id;
        e.price = p;
        e.qty = q;
        return e;
    }
    bool operator==(const InputEvent&) const = default;
};

// ---------------------------------------------------------------- outputs

enum class OutputKind : std::uint8_t {
    Accepted = 0, // new order passed validation
    Rejected,     // input refused, book unchanged
    Trade,        // one fill between a resting maker and the incoming taker
    Canceled,     // order left the book (or never rested) without filling
    Modified,     // modify applied; price/qty are the new values
    BookUpdate,   // L2: aggregate quantity at (side, price) is now `qty`
};

enum class RejectReason : std::uint8_t {
    None = 0,
    InvalidId,   // id 0 is reserved
    DuplicateId, // id is already resting in the book
    UnknownId,
    InvalidQty,
    InvalidPrice,
    PostOnlyWouldCross,
};

enum class CancelReason : std::uint8_t {
    None = 0,
    User,        // explicit cancel request
    Unfilled,    // IOC or Market remainder
    FokUnfilled, // FOK could not fill completely
};

struct OutputEvent {
    OutputKind kind{OutputKind::Accepted};
    SeqNo seq{0};          // input event that caused this output
    OrderId id{0};         // Accepted/Rejected/Canceled/Modified; taker for Trade
    OrderId maker_id{0};   // Trade only
    TradeId trade_id{0};   // Trade only
    Side side{Side::Buy};  // Trade: aggressor side. BookUpdate: book side
    Price price{0};
    Qty qty{0};            // Trade: fill qty. Canceled: qty removed. BookUpdate: new level total
    RejectReason reject{RejectReason::None};
    CancelReason cancel{CancelReason::None};

    bool operator==(const OutputEvent&) const = default;
};

const char* to_string(InputKind k);
const char* to_string(OutputKind k);
const char* to_string(RejectReason r);
const char* to_string(CancelReason r);

// One line, stable text encodings. These are the log format.
std::string encode(const InputEvent& e);
std::string encode(const OutputEvent& e);
bool decode(const std::string& line, InputEvent& out);

} // namespace exch
