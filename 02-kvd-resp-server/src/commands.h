// Command execution: argv in, RESP reply appended to an output buffer.
//
// The processor knows nothing about sockets or files. Write commands report
// themselves through the `propagate` hook in a *deterministic* form: relative
// expiries (SET EX/PX, EXPIRE, PEXPIRE) are rewritten to absolute ones
// (SET ... PXAT, PEXPIREAT) so that replaying the log later, at a different
// wall-clock time, reproduces the same expiry instant. The server wires this
// hook to the append-only file; tests wire it to a vector.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "store.h"

namespace kv {

class CommandProcessor {
public:
    using PropagateFn = std::function<void(const std::vector<std::string>&)>;

    enum class Action { None, Close };  // Close: flush the reply, then close the connection (QUIT)

    explicit CommandProcessor(Store& store) : store_(store) {}

    void set_propagate(PropagateFn fn) { propagate_ = std::move(fn); }

    // Execute one command. `args` must be non-empty. `now_ms` is Unix time in ms.
    Action execute(const std::vector<std::string>& args, int64_t now_ms, std::string& out);

    uint64_t commands_processed() const { return processed_; }

private:
    struct Ctx {
        const std::vector<std::string>& args;
        int64_t now;
        std::string& out;
    };
    using Handler = void (CommandProcessor::*)(Ctx&);
    struct Spec {
        const char* name;
        int arity;  // > 0: exact argc; < 0: at least -arity
        Handler fn;
    };
    static const Spec* lookup(std::string_view lower_name);

    void propagate(const std::vector<std::string>& argv) {
        if (propagate_) propagate_(argv);
    }
    void expire_at(Ctx& c, int64_t at_ms);
    void incr_generic(Ctx& c, int64_t delta);

    void cmd_ping(Ctx&);
    void cmd_echo(Ctx&);
    void cmd_get(Ctx&);
    void cmd_set(Ctx&);
    void cmd_del(Ctx&);
    void cmd_exists(Ctx&);
    void cmd_expire(Ctx&);
    void cmd_pexpire(Ctx&);
    void cmd_pexpireat(Ctx&);
    void cmd_persist(Ctx&);
    void cmd_ttl(Ctx&);
    void cmd_pttl(Ctx&);
    void cmd_incr(Ctx&);
    void cmd_decr(Ctx&);
    void cmd_incrby(Ctx&);
    void cmd_decrby(Ctx&);
    void cmd_keys(Ctx&);
    void cmd_dbsize(Ctx&);
    void cmd_flushall(Ctx&);
    void cmd_command(Ctx&);
    void cmd_config(Ctx&);
    void cmd_select(Ctx&);

    static const Spec kTable[];

    Store& store_;
    PropagateFn propagate_;
    uint64_t processed_ = 0;
};

}  // namespace kv
