// RESP2 request parsing and reply encoding.
//
// The parser is incremental in the simplest possible way: it is handed the
// unconsumed tail of a connection's input buffer and either returns one full
// command (and how many bytes it used), says "need more bytes", or reports a
// protocol error. It never keeps state between calls, so a partially received
// command is simply re-parsed from its start when more bytes arrive. Headers
// are tiny, and bulk payloads are only copied once the whole payload is present,
// so the re-parse cost is proportional to the number of arguments, not bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kv {

enum class ParseStatus { Ok, Incomplete, Error };

struct ParseResult {
    ParseStatus status = ParseStatus::Incomplete;
    size_t consumed = 0;     // bytes used when status == Ok
    std::string error;       // human readable message when status == Error
};

// Protocol limits, the same values real Redis uses by default.
inline constexpr int64_t kMaxBulkLen = 512LL * 1024 * 1024;     // proto-max-bulk-len
inline constexpr int64_t kMaxMultibulkLen = 1024 * 1024;        // max args per command
inline constexpr size_t kMaxInlineLen = 64 * 1024;              // inline command line / header line

// Parse one request from `buf`. Accepts both the multibulk form
// (`*N\r\n$len\r\narg\r\n...`, what every real client sends) and the inline
// form (`SET k v\r\n`, what you type into telnet or nc).
// On Ok, `args` holds the arguments. An empty `args` with Ok means an empty
// command (blank inline line or `*0`); the caller should skip it.
ParseResult parse_request(std::string_view buf, std::vector<std::string>& args);

// Strict signed 64 bit integer parse (no spaces, no '+', no leading zeros
// check, overflow rejected). Used for protocol lengths and command arguments.
bool parse_int64(std::string_view s, int64_t& out);

// Reply encoders. They append to `out`, which is a connection's output buffer.
void reply_simple(std::string& out, std::string_view s);   // +OK
void reply_error(std::string& out, std::string_view msg);  // -ERR ...
void reply_int(std::string& out, int64_t v);               // :42
void reply_bulk(std::string& out, std::string_view s);     // $3\r\nfoo
void reply_null(std::string& out);                         // $-1
void reply_array_header(std::string& out, size_t n);       // *n

// Encode argv as a RESP multibulk array. Used by the AOF and by clients.
void encode_command(std::string& out, const std::vector<std::string>& argv);

}  // namespace kv
