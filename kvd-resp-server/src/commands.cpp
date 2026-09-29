#include "commands.h"

#include <algorithm>
#include <cctype>
#include <limits>

#include "resp.h"

namespace kv {

namespace {

std::string to_lower(std::string_view s) {
    std::string r(s);
    for (char& ch : r) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return r;
}

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    return true;
}

constexpr const char* kErrNotInt = "ERR value is not an integer or out of range";
constexpr const char* kErrSyntax = "ERR syntax error";

// a + b with overflow detection.
bool add_overflows(int64_t a, int64_t b, int64_t& out) { return __builtin_add_overflow(a, b, &out); }
bool mul_overflows(int64_t a, int64_t b, int64_t& out) { return __builtin_mul_overflow(a, b, &out); }

}  // namespace

// Hot commands first: lookup is a linear scan over ~25 entries after
// lowercasing. A hash map would win only with many more commands.
const CommandProcessor::Spec CommandProcessor::kTable[] = {
    {"get", 2, &CommandProcessor::cmd_get},
    {"set", -3, &CommandProcessor::cmd_set},
    {"ping", -1, &CommandProcessor::cmd_ping},
    {"del", -2, &CommandProcessor::cmd_del},
    {"exists", -2, &CommandProcessor::cmd_exists},
    {"incr", 2, &CommandProcessor::cmd_incr},
    {"decr", 2, &CommandProcessor::cmd_decr},
    {"incrby", 3, &CommandProcessor::cmd_incrby},
    {"decrby", 3, &CommandProcessor::cmd_decrby},
    {"expire", 3, &CommandProcessor::cmd_expire},
    {"pexpire", 3, &CommandProcessor::cmd_pexpire},
    {"pexpireat", 3, &CommandProcessor::cmd_pexpireat},
    {"persist", 2, &CommandProcessor::cmd_persist},
    {"ttl", 2, &CommandProcessor::cmd_ttl},
    {"pttl", 2, &CommandProcessor::cmd_pttl},
    {"keys", 2, &CommandProcessor::cmd_keys},
    {"echo", 2, &CommandProcessor::cmd_echo},
    {"dbsize", 1, &CommandProcessor::cmd_dbsize},
    {"flushall", -1, &CommandProcessor::cmd_flushall},
    {"flushdb", -1, &CommandProcessor::cmd_flushall},
    {"command", -1, &CommandProcessor::cmd_command},
    {"config", -2, &CommandProcessor::cmd_config},
    {"select", 2, &CommandProcessor::cmd_select},
};

const CommandProcessor::Spec* CommandProcessor::lookup(std::string_view lower) {
    for (const Spec& s : kTable)
        if (lower == s.name) return &s;
    return nullptr;
}

CommandProcessor::Action CommandProcessor::execute(const std::vector<std::string>& args, int64_t now_ms,
                                                   std::string& out) {
    ++processed_;
    std::string name = to_lower(args[0]);
    if (name == "quit") {
        reply_simple(out, "OK");
        return Action::Close;
    }
    const Spec* spec = lookup(name);
    if (!spec) {
        std::string msg = "ERR unknown command '" + args[0] + "', with args beginning with: ";
        for (size_t i = 1; i < args.size() && i < 4; ++i) msg += "'" + args[i] + "' ";
        reply_error(out, msg);
        return Action::None;
    }
    int argc = static_cast<int>(args.size());
    if ((spec->arity > 0 && argc != spec->arity) || (spec->arity < 0 && argc < -spec->arity)) {
        reply_error(out, "ERR wrong number of arguments for '" + name + "' command");
        return Action::None;
    }
    Ctx c{args, now_ms, out};
    (this->*(spec->fn))(c);
    return Action::None;
}

void CommandProcessor::cmd_ping(Ctx& c) {
    if (c.args.size() > 2) {
        reply_error(c.out, "ERR wrong number of arguments for 'ping' command");
    } else if (c.args.size() == 2) {
        reply_bulk(c.out, c.args[1]);
    } else {
        reply_simple(c.out, "PONG");
    }
}

void CommandProcessor::cmd_echo(Ctx& c) { reply_bulk(c.out, c.args[1]); }

void CommandProcessor::cmd_get(Ctx& c) {
    Entry* e = store_.find(c.args[1], c.now);
    if (e)
        reply_bulk(c.out, e->value);
    else
        reply_null(c.out);
}

// SET key value [NX | XX] [EX s | PX ms | EXAT s | PXAT ms | KEEPTTL]
void CommandProcessor::cmd_set(Ctx& c) {
    bool nx = false, xx = false, keep_ttl = false;
    int64_t expire_at = kNoExpiry;
    bool have_expire = false;
    for (size_t i = 3; i < c.args.size(); ++i) {
        const std::string& opt = c.args[i];
        bool has_next = i + 1 < c.args.size();
        if (iequals(opt, "nx") && !xx) {
            nx = true;
        } else if (iequals(opt, "xx") && !nx) {
            xx = true;
        } else if (iequals(opt, "keepttl") && !have_expire) {
            keep_ttl = true;
        } else if ((iequals(opt, "ex") || iequals(opt, "px") || iequals(opt, "exat") || iequals(opt, "pxat")) &&
                   has_next && !have_expire && !keep_ttl) {
            int64_t v;
            if (!parse_int64(c.args[i + 1], v)) {
                reply_error(c.out, kErrNotInt);
                return;
            }
            bool seconds = iequals(opt, "ex") || iequals(opt, "exat");
            bool relative = iequals(opt, "ex") || iequals(opt, "px");
            int64_t ms = v;
            if (v <= 0 || (seconds && mul_overflows(v, 1000, ms)) || (relative && add_overflows(ms, c.now, ms))) {
                reply_error(c.out, "ERR invalid expire time in 'set' command");
                return;
            }
            expire_at = ms;
            have_expire = true;
            ++i;
        } else {
            reply_error(c.out, kErrSyntax);
            return;
        }
    }
    const std::string& key = c.args[1];
    if (nx || xx) {
        bool exists = store_.find(key, c.now) != nullptr;
        if ((nx && exists) || (xx && !exists)) {
            reply_null(c.out);
            return;
        }
    }
    store_.set(key, c.args[2], expire_at, keep_ttl);
    reply_simple(c.out, "OK");

    if (propagate_) {
        std::vector<std::string> argv{"SET", key, c.args[2]};
        if (have_expire) {
            argv.push_back("PXAT");
            argv.push_back(std::to_string(expire_at));
        } else if (keep_ttl) {
            argv.push_back("KEEPTTL");
        }
        propagate(argv);
    }
}

void CommandProcessor::cmd_del(Ctx& c) {
    int64_t n = 0;
    for (size_t i = 1; i < c.args.size(); ++i) n += store_.del(c.args[i], c.now) ? 1 : 0;
    reply_int(c.out, n);
    if (n > 0) propagate(c.args);
}

void CommandProcessor::cmd_exists(Ctx& c) {
    int64_t n = 0;
    for (size_t i = 1; i < c.args.size(); ++i) n += store_.find(c.args[i], c.now) ? 1 : 0;
    reply_int(c.out, n);
}

// Shared tail of EXPIRE / PEXPIRE / PEXPIREAT once the absolute deadline is known.
void CommandProcessor::expire_at(Ctx& c, int64_t at_ms) {
    const std::string& key = c.args[1];
    if (at_ms <= c.now) {
        // A deadline in the past deletes the key right away, as in Redis, and
        // is logged as a DEL so replay does not depend on the replay-time clock.
        bool existed = store_.del(key, c.now);
        reply_int(c.out, existed ? 1 : 0);
        if (existed) propagate({"DEL", key});
        return;
    }
    bool ok = store_.set_expiry(key, at_ms, c.now);
    reply_int(c.out, ok ? 1 : 0);
    if (ok) propagate({"PEXPIREAT", key, std::to_string(at_ms)});
}

void CommandProcessor::cmd_expire(Ctx& c) {
    int64_t secs, ms, at;
    if (!parse_int64(c.args[2], secs)) return reply_error(c.out, kErrNotInt);
    if (mul_overflows(secs, 1000, ms) || add_overflows(ms, c.now, at))
        return reply_error(c.out, "ERR invalid expire time in 'expire' command");
    expire_at(c, at);
}

void CommandProcessor::cmd_pexpire(Ctx& c) {
    int64_t ms, at;
    if (!parse_int64(c.args[2], ms)) return reply_error(c.out, kErrNotInt);
    if (add_overflows(ms, c.now, at)) return reply_error(c.out, "ERR invalid expire time in 'pexpire' command");
    expire_at(c, at);
}

void CommandProcessor::cmd_pexpireat(Ctx& c) {
    int64_t at;
    if (!parse_int64(c.args[2], at)) return reply_error(c.out, kErrNotInt);
    expire_at(c, at);
}

void CommandProcessor::cmd_persist(Ctx& c) {
    Entry* e = store_.find(c.args[1], c.now);
    if (!e || e->expire_at_ms == kNoExpiry) return reply_int(c.out, 0);
    store_.set_expiry(c.args[1], kNoExpiry, c.now);
    reply_int(c.out, 1);
    propagate(c.args);
}

void CommandProcessor::cmd_ttl(Ctx& c) {
    int64_t ms = store_.pttl(c.args[1], c.now);
    reply_int(c.out, ms < 0 ? ms : (ms + 500) / 1000);  // Redis rounds to the nearest second
}

void CommandProcessor::cmd_pttl(Ctx& c) { reply_int(c.out, store_.pttl(c.args[1], c.now)); }

void CommandProcessor::incr_generic(Ctx& c, int64_t delta) {
    const std::string& key = c.args[1];
    Entry* e = store_.find(key, c.now);
    int64_t cur = 0;
    if (e && !parse_int64(e->value, cur)) return reply_error(c.out, kErrNotInt);
    int64_t next;
    if (add_overflows(cur, delta, next)) return reply_error(c.out, "ERR increment or decrement would overflow");
    if (e)
        e->value = std::to_string(next);  // in place: keeps the TTL, as Redis does
    else
        store_.set(key, std::to_string(next), kNoExpiry);
    reply_int(c.out, next);
    propagate(c.args);  // INCR/INCRBY are deterministic given the prior state
}

void CommandProcessor::cmd_incr(Ctx& c) { incr_generic(c, 1); }
void CommandProcessor::cmd_decr(Ctx& c) { incr_generic(c, -1); }

void CommandProcessor::cmd_incrby(Ctx& c) {
    int64_t d;
    if (!parse_int64(c.args[2], d)) return reply_error(c.out, kErrNotInt);
    incr_generic(c, d);
}

void CommandProcessor::cmd_decrby(Ctx& c) {
    int64_t d;
    if (!parse_int64(c.args[2], d) || d == std::numeric_limits<int64_t>::min())
        return reply_error(c.out, kErrNotInt);
    incr_generic(c, -d);
}

void CommandProcessor::cmd_keys(Ctx& c) {
    std::vector<std::string> ks;
    store_.keys(c.args[1], c.now, ks);
    reply_array_header(c.out, ks.size());
    for (const auto& k : ks) reply_bulk(c.out, k);
}

void CommandProcessor::cmd_dbsize(Ctx& c) { reply_int(c.out, static_cast<int64_t>(store_.size())); }

void CommandProcessor::cmd_flushall(Ctx& c) {
    store_.clear();
    reply_simple(c.out, "OK");
    propagate({"FLUSHALL"});
}

// redis-cli sends COMMAND DOCS on startup to build its hints; an empty array
// is a valid answer and makes it fall back to no hints.
void CommandProcessor::cmd_command(Ctx& c) { reply_array_header(c.out, 0); }

// redis-benchmark asks CONFIG GET save / appendonly. We answer with an empty
// map (no such parameters) rather than an error.
void CommandProcessor::cmd_config(Ctx& c) {
    if (iequals(c.args[1], "get")) return reply_array_header(c.out, 0);
    reply_error(c.out, "ERR CONFIG subcommand not supported");
}

void CommandProcessor::cmd_select(Ctx& c) {
    if (c.args[1] == "0") return reply_simple(c.out, "OK");
    reply_error(c.out, "ERR DB index is out of range");
}

}  // namespace kv
