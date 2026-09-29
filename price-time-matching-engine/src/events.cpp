#include "exchange/events.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace exch {

// ------------------------------------------------------------ fixed point

std::optional<std::int64_t> parse_fixed(std::string_view text, int scale) {
    // Accepts [-]digits[.digits] with at most `scale` fractional digits.
    // Rejects anything that would need rounding: prices are exact.
    if (text.empty() || scale < 0 || scale > 9) return std::nullopt;
    bool neg = false;
    std::size_t i = 0;
    if (text[0] == '-') { neg = true; i = 1; }
    std::int64_t whole = 0, frac = 0;
    int frac_digits = 0;
    bool seen_dot = false, any_digit = false;
    for (; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '.') {
            if (seen_dot) return std::nullopt;
            seen_dot = true;
            continue;
        }
        if (c < '0' || c > '9') return std::nullopt;
        any_digit = true;
        if (seen_dot) {
            if (++frac_digits > scale) return std::nullopt;
            frac = frac * 10 + (c - '0');
        } else {
            if (whole > (INT64_MAX / 10) - 10) return std::nullopt;
            whole = whole * 10 + (c - '0');
        }
    }
    if (!any_digit) return std::nullopt;
    std::int64_t mult = 1;
    for (int k = 0; k < scale; ++k) mult *= 10;
    for (int k = frac_digits; k < scale; ++k) frac *= 10;
    if (whole > (INT64_MAX - frac) / mult) return std::nullopt;
    const std::int64_t v = whole * mult + frac;
    return neg ? -v : v;
}

std::string format_fixed(std::int64_t value, int scale) {
    std::int64_t mult = 1;
    for (int k = 0; k < scale; ++k) mult *= 10;
    const bool neg = value < 0;
    const std::uint64_t mag = neg ? 0 - static_cast<std::uint64_t>(value) : static_cast<std::uint64_t>(value);
    const std::uint64_t whole = mag / static_cast<std::uint64_t>(mult);
    const std::uint64_t frac = mag % static_cast<std::uint64_t>(mult);
    char buf[64];
    if (scale == 0)
        std::snprintf(buf, sizeof buf, "%s%" PRIu64, neg ? "-" : "", whole);
    else
        std::snprintf(buf, sizeof buf, "%s%" PRIu64 ".%0*" PRIu64, neg ? "-" : "", whole, scale, frac);
    return buf;
}

// ------------------------------------------------------------- names

const char* to_string(Side s) { return s == Side::Buy ? "B" : "S"; }

const char* to_string(OrderType t) {
    switch (t) {
    case OrderType::Limit: return "LMT";
    case OrderType::Market: return "MKT";
    case OrderType::IOC: return "IOC";
    case OrderType::FOK: return "FOK";
    case OrderType::PostOnly: return "POST";
    }
    return "?";
}

const char* to_string(InputKind k) {
    switch (k) {
    case InputKind::New: return "N";
    case InputKind::Cancel: return "C";
    case InputKind::Modify: return "M";
    }
    return "?";
}

const char* to_string(OutputKind k) {
    switch (k) {
    case OutputKind::Accepted: return "ACCEPTED";
    case OutputKind::Rejected: return "REJECTED";
    case OutputKind::Trade: return "TRADE";
    case OutputKind::Canceled: return "CANCELED";
    case OutputKind::Modified: return "MODIFIED";
    case OutputKind::BookUpdate: return "L2";
    }
    return "?";
}

const char* to_string(RejectReason r) {
    switch (r) {
    case RejectReason::None: return "none";
    case RejectReason::InvalidId: return "invalid_id";
    case RejectReason::DuplicateId: return "duplicate_id";
    case RejectReason::UnknownId: return "unknown_id";
    case RejectReason::InvalidQty: return "invalid_qty";
    case RejectReason::InvalidPrice: return "invalid_price";
    case RejectReason::PostOnlyWouldCross: return "post_only_would_cross";
    }
    return "?";
}

const char* to_string(CancelReason r) {
    switch (r) {
    case CancelReason::None: return "none";
    case CancelReason::User: return "user";
    case CancelReason::Unfilled: return "unfilled";
    case CancelReason::FokUnfilled: return "fok_unfilled";
    }
    return "?";
}

// ------------------------------------------------------------- encoding

std::string encode(const InputEvent& e) {
    char buf[160];
    switch (e.kind) {
    case InputKind::New:
        std::snprintf(buf, sizeof buf, "N %" PRIu64 " %s %s %" PRId64 " %" PRId64, e.id,
                      to_string(e.side), to_string(e.type), e.price, e.qty);
        break;
    case InputKind::Cancel: std::snprintf(buf, sizeof buf, "C %" PRIu64, e.id); break;
    case InputKind::Modify:
        std::snprintf(buf, sizeof buf, "M %" PRIu64 " %" PRId64 " %" PRId64, e.id, e.price, e.qty);
        break;
    }
    return buf;
}

std::string encode(const OutputEvent& e) {
    char buf[200];
    switch (e.kind) {
    case OutputKind::Accepted:
        std::snprintf(buf, sizeof buf, "%" PRIu64 " ACCEPTED %" PRIu64 " %s %" PRId64 " %" PRId64, e.seq, e.id,
                      to_string(e.side), e.price, e.qty);
        break;
    case OutputKind::Rejected:
        std::snprintf(buf, sizeof buf, "%" PRIu64 " REJECTED %" PRIu64 " %s", e.seq, e.id, to_string(e.reject));
        break;
    case OutputKind::Trade:
        std::snprintf(buf, sizeof buf,
                      "%" PRIu64 " TRADE %" PRIu64 " taker=%" PRIu64 " maker=%" PRIu64 " %s %" PRId64 " %" PRId64,
                      e.seq, e.trade_id, e.id, e.maker_id, to_string(e.side), e.price, e.qty);
        break;
    case OutputKind::Canceled:
        std::snprintf(buf, sizeof buf, "%" PRIu64 " CANCELED %" PRIu64 " %s %" PRId64 " %" PRId64 " %s", e.seq, e.id,
                      to_string(e.side), e.price, e.qty, to_string(e.cancel));
        break;
    case OutputKind::Modified:
        std::snprintf(buf, sizeof buf, "%" PRIu64 " MODIFIED %" PRIu64 " %s %" PRId64 " %" PRId64, e.seq, e.id,
                      to_string(e.side), e.price, e.qty);
        break;
    case OutputKind::BookUpdate:
        std::snprintf(buf, sizeof buf, "%" PRIu64 " L2 %s %" PRId64 " %" PRId64, e.seq, to_string(e.side), e.price,
                      e.qty);
        break;
    }
    return buf;
}

bool decode(const std::string& line, InputEvent& out) {
    std::istringstream is(line);
    std::string tag;
    if (!(is >> tag)) return false;
    out = InputEvent{};
    if (tag == "N") {
        std::string side, type;
        if (!(is >> out.id >> side >> type >> out.price >> out.qty)) return false;
        out.kind = InputKind::New;
        if (side == "B") out.side = Side::Buy;
        else if (side == "S") out.side = Side::Sell;
        else return false;
        if (type == "LMT") out.type = OrderType::Limit;
        else if (type == "MKT") out.type = OrderType::Market;
        else if (type == "IOC") out.type = OrderType::IOC;
        else if (type == "FOK") out.type = OrderType::FOK;
        else if (type == "POST") out.type = OrderType::PostOnly;
        else return false;
        return true;
    }
    if (tag == "C") {
        out.kind = InputKind::Cancel;
        return static_cast<bool>(is >> out.id);
    }
    if (tag == "M") {
        out.kind = InputKind::Modify;
        return static_cast<bool>(is >> out.id >> out.price >> out.qty);
    }
    return false;
}

} // namespace exch
