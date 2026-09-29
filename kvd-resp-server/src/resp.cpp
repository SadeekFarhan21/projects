#include "resp.h"

#include <charconv>
#include <cstring>

namespace kv {

bool parse_int64(std::string_view s, int64_t& out) {
    if (s.empty() || s.size() > 20) return false;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc() && ptr == last;
}

namespace {

// Find "\r\n" starting at `from`. Returns npos when absent.
size_t find_crlf(std::string_view buf, size_t from) {
    while (from < buf.size()) {
        const void* p = std::memchr(buf.data() + from, '\r', buf.size() - from);
        if (!p) return std::string_view::npos;
        size_t i = static_cast<const char*>(p) - buf.data();
        if (i + 1 >= buf.size()) return std::string_view::npos;  // '\r' is the last byte, wait
        if (buf[i + 1] == '\n') return i;
        from = i + 1;
    }
    return std::string_view::npos;
}

ParseResult make_error(std::string msg) {
    ParseResult r;
    r.status = ParseStatus::Error;
    r.error = std::move(msg);
    return r;
}

ParseResult parse_inline(std::string_view buf, std::vector<std::string>& args) {
    size_t nl = buf.find('\n');
    if (nl == std::string_view::npos) {
        if (buf.size() > kMaxInlineLen) return make_error("Protocol error: too big inline request");
        return {};
    }
    std::string_view line = buf.substr(0, nl);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    args.clear();
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        size_t start = i;
        while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
        if (i > start) args.emplace_back(line.substr(start, i - start));
    }
    ParseResult r;
    r.status = ParseStatus::Ok;
    r.consumed = nl + 1;
    return r;
}

ParseResult parse_multibulk(std::string_view buf, std::vector<std::string>& args) {
    size_t eol = find_crlf(buf, 0);
    if (eol == std::string_view::npos) {
        if (buf.size() > kMaxInlineLen) return make_error("Protocol error: too big mbulk count string");
        return {};
    }
    int64_t n = 0;
    if (!parse_int64(buf.substr(1, eol - 1), n) || n > kMaxMultibulkLen)
        return make_error("Protocol error: invalid multibulk length");
    size_t pos = eol + 2;
    if (n <= 0) {  // *0 and *-1 are empty commands; skip them like Redis does
        args.clear();
        return {ParseStatus::Ok, pos, {}};
    }

    // First pass: validate the whole frame is present without copying payloads.
    // This keeps a slowly arriving large command from being copied repeatedly.
    struct Span { size_t off, len; };
    std::vector<Span> spans;
    spans.reserve(static_cast<size_t>(n));
    for (int64_t k = 0; k < n; ++k) {
        if (pos >= buf.size()) return {};
        if (buf[pos] != '$')
            return make_error(std::string("Protocol error: expected '$', got '") + buf[pos] + "'");
        size_t e = find_crlf(buf, pos);
        if (e == std::string_view::npos) {
            if (buf.size() - pos > kMaxInlineLen) return make_error("Protocol error: too big bulk count string");
            return {};
        }
        int64_t len = 0;
        if (!parse_int64(buf.substr(pos + 1, e - pos - 1), len) || len < 0 || len > kMaxBulkLen)
            return make_error("Protocol error: invalid bulk length");
        size_t data = e + 2;
        size_t need = data + static_cast<size_t>(len) + 2;
        if (buf.size() < need) return {};
        if (buf[need - 2] != '\r' || buf[need - 1] != '\n')
            return make_error("Protocol error: bulk string not terminated by CRLF");
        spans.push_back({data, static_cast<size_t>(len)});
        pos = need;
    }
    args.resize(spans.size());
    for (size_t k = 0; k < spans.size(); ++k) args[k].assign(buf.data() + spans[k].off, spans[k].len);
    return {ParseStatus::Ok, pos, {}};
}

}  // namespace

ParseResult parse_request(std::string_view buf, std::vector<std::string>& args) {
    if (buf.empty()) return {};
    if (buf[0] == '*') return parse_multibulk(buf, args);
    return parse_inline(buf, args);
}

void reply_simple(std::string& out, std::string_view s) {
    out.push_back('+');
    out.append(s);
    out.append("\r\n");
}

void reply_error(std::string& out, std::string_view msg) {
    out.push_back('-');
    out.append(msg);
    out.append("\r\n");
}

static void append_int(std::string& out, int64_t v) {
    char tmp[24];
    auto [p, ec] = std::to_chars(tmp, tmp + sizeof(tmp), v);
    (void)ec;
    out.append(tmp, p);
}

void reply_int(std::string& out, int64_t v) {
    out.push_back(':');
    append_int(out, v);
    out.append("\r\n");
}

void reply_bulk(std::string& out, std::string_view s) {
    out.push_back('$');
    append_int(out, static_cast<int64_t>(s.size()));
    out.append("\r\n");
    out.append(s);
    out.append("\r\n");
}

void reply_null(std::string& out) { out.append("$-1\r\n"); }

void reply_array_header(std::string& out, size_t n) {
    out.push_back('*');
    append_int(out, static_cast<int64_t>(n));
    out.append("\r\n");
}

void encode_command(std::string& out, const std::vector<std::string>& argv) {
    reply_array_header(out, argv.size());
    for (const auto& a : argv) reply_bulk(out, a);
}

}  // namespace kv
