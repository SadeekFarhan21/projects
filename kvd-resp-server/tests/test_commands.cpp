#include <gtest/gtest.h>

#include "commands.h"
#include "resp.h"

using namespace kv;

namespace {

class Cmd : public ::testing::Test {
protected:
    Cmd() : proc(store) {
        proc.set_propagate([this](const std::vector<std::string>& a) { log.push_back(a); });
    }
    // Run a command at the current fake time and return the raw RESP reply.
    std::string run(std::vector<std::string> args) {
        std::string out;
        proc.execute(args, now, out);
        return out;
    }
    Store store;
    CommandProcessor proc;
    std::vector<std::vector<std::string>> log;
    int64_t now = 1'700'000'000'000;  // a realistic unix-ms timestamp
};

}  // namespace

TEST_F(Cmd, PingEcho) {
    EXPECT_EQ(run({"PING"}), "+PONG\r\n");
    EXPECT_EQ(run({"ping", "hi"}), "$2\r\nhi\r\n");
    EXPECT_EQ(run({"ECHO", "x"}), "$1\r\nx\r\n");
    EXPECT_EQ(run({"PING", "a", "b"}), "-ERR wrong number of arguments for 'ping' command\r\n");
}

TEST_F(Cmd, SetGetDelExists) {
    EXPECT_EQ(run({"GET", "k"}), "$-1\r\n");
    EXPECT_EQ(run({"SET", "k", "v"}), "+OK\r\n");
    EXPECT_EQ(run({"get", "k"}), "$1\r\nv\r\n");
    EXPECT_EQ(run({"EXISTS", "k", "k", "nope"}), ":2\r\n");
    EXPECT_EQ(run({"DEL", "k", "nope"}), ":1\r\n");
    EXPECT_EQ(run({"DEL", "k"}), ":0\r\n");
    EXPECT_EQ(run({"EXISTS", "k"}), ":0\r\n");
}

TEST_F(Cmd, ArityAndUnknown) {
    EXPECT_EQ(run({"GET"}), "-ERR wrong number of arguments for 'get' command\r\n");
    EXPECT_EQ(run({"SET", "k"}), "-ERR wrong number of arguments for 'set' command\r\n");
    EXPECT_EQ(run({"NOPE", "a"}).rfind("-ERR unknown command 'NOPE'", 0), 0u);
}

TEST_F(Cmd, SetOptions) {
    EXPECT_EQ(run({"SET", "k", "v", "EX", "10"}), "+OK\r\n");
    EXPECT_EQ(run({"PTTL", "k"}), ":10000\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "PX", "1500"}), "+OK\r\n");
    EXPECT_EQ(run({"TTL", "k"}), ":2\r\n");  // 1.5 s rounds to 2, like Redis
    EXPECT_EQ(run({"SET", "k", "v2", "KEEPTTL"}), "+OK\r\n");
    EXPECT_EQ(run({"PTTL", "k"}), ":1500\r\n");
    EXPECT_EQ(run({"SET", "k", "v3"}), "+OK\r\n");
    EXPECT_EQ(run({"TTL", "k"}), ":-1\r\n");

    EXPECT_EQ(run({"SET", "n", "v", "NX"}), "+OK\r\n");
    EXPECT_EQ(run({"SET", "n", "w", "NX"}), "$-1\r\n");
    EXPECT_EQ(run({"GET", "n"}), "$1\r\nv\r\n");
    EXPECT_EQ(run({"SET", "x", "v", "XX"}), "$-1\r\n");
    EXPECT_EQ(run({"SET", "n", "w", "XX"}), "+OK\r\n");

    EXPECT_EQ(run({"SET", "k", "v", "EX", "0"}), "-ERR invalid expire time in 'set' command\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "EX", "-5"}), "-ERR invalid expire time in 'set' command\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "EX", "abc"}), "-ERR value is not an integer or out of range\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "EX", "9223372036854775807"}),
              "-ERR invalid expire time in 'set' command\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "NX", "XX"}), "-ERR syntax error\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "EX", "1", "PX", "1"}), "-ERR syntax error\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "EX"}), "-ERR syntax error\r\n");
    EXPECT_EQ(run({"SET", "k", "v", "KEEPTTL", "EX", "1"}), "-ERR syntax error\r\n");
}

TEST_F(Cmd, ExpiryIsObservedThroughTime) {
    run({"SET", "k", "v", "PX", "100"});
    now += 99;
    EXPECT_EQ(run({"GET", "k"}), "$1\r\nv\r\n");
    now += 1;
    EXPECT_EQ(run({"GET", "k"}), "$-1\r\n");
    EXPECT_EQ(run({"TTL", "k"}), ":-2\r\n");
}

TEST_F(Cmd, ExpireTtlPersist) {
    EXPECT_EQ(run({"EXPIRE", "k", "10"}), ":0\r\n");
    run({"SET", "k", "v"});
    EXPECT_EQ(run({"TTL", "k"}), ":-1\r\n");
    EXPECT_EQ(run({"EXPIRE", "k", "10"}), ":1\r\n");
    EXPECT_EQ(run({"TTL", "k"}), ":10\r\n");
    EXPECT_EQ(run({"PEXPIRE", "k", "2500"}), ":1\r\n");
    EXPECT_EQ(run({"PTTL", "k"}), ":2500\r\n");
    EXPECT_EQ(run({"PERSIST", "k"}), ":1\r\n");
    EXPECT_EQ(run({"PERSIST", "k"}), ":0\r\n");
    EXPECT_EQ(run({"TTL", "k"}), ":-1\r\n");
    EXPECT_EQ(run({"EXPIRE", "k", "x"}), "-ERR value is not an integer or out of range\r\n");
    EXPECT_EQ(run({"EXPIRE", "k", "9223372036854775807"}), "-ERR invalid expire time in 'expire' command\r\n");
    // Non-positive TTL deletes immediately.
    EXPECT_EQ(run({"EXPIRE", "k", "0"}), ":1\r\n");
    EXPECT_EQ(run({"EXISTS", "k"}), ":0\r\n");
}

TEST_F(Cmd, Incr) {
    EXPECT_EQ(run({"INCR", "c"}), ":1\r\n");
    EXPECT_EQ(run({"INCR", "c"}), ":2\r\n");
    EXPECT_EQ(run({"INCRBY", "c", "40"}), ":42\r\n");
    EXPECT_EQ(run({"DECR", "c"}), ":41\r\n");
    EXPECT_EQ(run({"DECRBY", "c", "41"}), ":0\r\n");
    run({"SET", "s", "abc"});
    EXPECT_EQ(run({"INCR", "s"}), "-ERR value is not an integer or out of range\r\n");
    run({"SET", "m", "9223372036854775807"});
    EXPECT_EQ(run({"INCR", "m"}), "-ERR increment or decrement would overflow\r\n");
    EXPECT_EQ(run({"GET", "m"}), "$19\r\n9223372036854775807\r\n");
    // INCR keeps an existing TTL.
    run({"SET", "t", "5", "EX", "100"});
    run({"INCR", "t"});
    EXPECT_EQ(run({"TTL", "t"}), ":100\r\n");
}

TEST_F(Cmd, Keys) {
    run({"SET", "user:1", "a"});
    run({"SET", "user:2", "b"});
    run({"SET", "other", "c"});
    std::string out = run({"KEYS", "user:*"});
    EXPECT_EQ(out.substr(0, 4), "*2\r\n");
    EXPECT_NE(out.find("user:1"), std::string::npos);
    EXPECT_NE(out.find("user:2"), std::string::npos);
    EXPECT_EQ(run({"KEYS", "zzz*"}), "*0\r\n");
    EXPECT_EQ(run({"DBSIZE"}), ":3\r\n");
}

TEST_F(Cmd, QuitAsksToClose) {
    std::string out;
    EXPECT_EQ(proc.execute({"QUIT"}, now, out), CommandProcessor::Action::Close);
    EXPECT_EQ(out, "+OK\r\n");
}

TEST_F(Cmd, PropagationIsDeterministic) {
    run({"SET", "a", "1", "EX", "10"});
    run({"SET", "b", "2"});
    run({"SET", "b", "3", "KEEPTTL"});
    run({"EXPIRE", "b", "5"});
    run({"PEXPIRE", "b", "-1"});  // past deadline: logged as DEL
    run({"INCR", "c"});
    run({"DEL", "nope"});         // no-op: not logged
    run({"GET", "a"});            // read: not logged
    run({"SET", "x", "v", "NX"});
    run({"SET", "x", "w", "NX"}); // rejected by NX: not logged
    const std::string at10 = std::to_string(now + 10000), at5 = std::to_string(now + 5000);
    std::vector<std::vector<std::string>> want = {
        {"SET", "a", "1", "PXAT", at10},
        {"SET", "b", "2"},
        {"SET", "b", "3", "KEEPTTL"},
        {"PEXPIREAT", "b", at5},
        {"DEL", "b"},
        {"INCR", "c"},
        {"SET", "x", "v"},
    };
    EXPECT_EQ(log, want);
}

TEST_F(Cmd, ReplayOfPropagatedLogReproducesState) {
    run({"SET", "a", "1", "EX", "100"});
    run({"SET", "b", "x"});
    run({"INCRBY", "n", "7"});
    run({"EXPIRE", "b", "50"});
    run({"DEL", "zz"});

    // Replay the log into a fresh store 30 s later: absolute deadlines mean
    // the remaining TTL shrinks by exactly the elapsed time.
    Store s2;
    CommandProcessor p2(s2);
    int64_t later = now + 30000;
    std::string sink;
    for (auto& a : log) p2.execute(a, later, sink);
    EXPECT_EQ(s2.pttl("a", later), 70000);
    EXPECT_EQ(s2.pttl("b", later), 20000);
    EXPECT_EQ(s2.find("n", later)->value, "7");
    EXPECT_EQ(s2.size(), 3u);
}
