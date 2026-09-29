#include <gtest/gtest.h>

#include "resp.h"

using namespace kv;

namespace {
std::string cmd(const std::vector<std::string>& a) {
    std::string s;
    encode_command(s, a);
    return s;
}
}  // namespace

TEST(Resp, ParsesMultibulk) {
    std::vector<std::string> args;
    std::string in = "*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n";
    auto r = parse_request(in, args);
    ASSERT_EQ(r.status, ParseStatus::Ok);
    EXPECT_EQ(r.consumed, in.size());
    EXPECT_EQ(args, (std::vector<std::string>{"SET", "key", "value"}));
}

TEST(Resp, BinarySafeBulk) {
    std::vector<std::string> args;
    std::string payload("a\r\n\0b", 5);
    std::string in = cmd({"SET", "k", payload});
    auto r = parse_request(in, args);
    ASSERT_EQ(r.status, ParseStatus::Ok);
    EXPECT_EQ(args[2], payload);
}

TEST(Resp, EveryPrefixIsIncomplete) {
    // Feeding the frame one byte at a time must never produce a false Ok or an error.
    std::string in = cmd({"SET", "hello", "world", "EX", "10"});
    std::vector<std::string> args;
    for (size_t n = 0; n < in.size(); ++n) {
        auto r = parse_request(std::string_view(in).substr(0, n), args);
        EXPECT_EQ(r.status, ParseStatus::Incomplete) << "prefix length " << n;
    }
    auto r = parse_request(in, args);
    EXPECT_EQ(r.status, ParseStatus::Ok);
}

TEST(Resp, PipelinedFramesParseOneAtATime) {
    std::string in = cmd({"PING"}) + cmd({"GET", "a"}) + cmd({"DEL", "a", "b"});
    std::vector<std::string> args;
    size_t pos = 0;
    std::vector<std::vector<std::string>> got;
    while (pos < in.size()) {
        auto r = parse_request(std::string_view(in).substr(pos), args);
        ASSERT_EQ(r.status, ParseStatus::Ok);
        got.push_back(args);
        pos += r.consumed;
    }
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[2], (std::vector<std::string>{"DEL", "a", "b"}));
}

TEST(Resp, InlineCommands) {
    std::vector<std::string> args;
    auto r = parse_request("SET  k \t v\r\nGET k\n", args);
    ASSERT_EQ(r.status, ParseStatus::Ok);
    EXPECT_EQ(args, (std::vector<std::string>{"SET", "k", "v"}));
    EXPECT_EQ(r.consumed, 12u);  // "SET  k \t v\r\n"
    r = parse_request("\r\n", args);
    ASSERT_EQ(r.status, ParseStatus::Ok);
    EXPECT_TRUE(args.empty());
    EXPECT_EQ(parse_request("PING", args).status, ParseStatus::Incomplete);
}

TEST(Resp, EmptyMultibulkIsSkippable) {
    std::vector<std::string> args{"junk"};
    auto r = parse_request("*0\r\n", args);
    ASSERT_EQ(r.status, ParseStatus::Ok);
    EXPECT_EQ(r.consumed, 4u);
    EXPECT_TRUE(args.empty());
}

TEST(Resp, ProtocolErrors) {
    std::vector<std::string> args;
    EXPECT_EQ(parse_request("*x\r\n", args).status, ParseStatus::Error);
    EXPECT_EQ(parse_request("*1\r\n+PING\r\n", args).status, ParseStatus::Error);
    EXPECT_EQ(parse_request("*1\r\n$-5\r\n", args).status, ParseStatus::Error);
    EXPECT_EQ(parse_request("*1\r\n$4\r\nPINGxx", args).status, ParseStatus::Error);
    EXPECT_EQ(parse_request("*99999999\r\n", args).status, ParseStatus::Error);
    EXPECT_EQ(parse_request("*1\r\n$999999999999\r\n", args).status, ParseStatus::Error);
    // A header with no CRLF that is already absurdly long is an error, not "wait forever".
    EXPECT_EQ(parse_request("*" + std::string(70000, '1'), args).status, ParseStatus::Error);
}

TEST(Resp, ReplyEncoding) {
    std::string out;
    reply_simple(out, "OK");
    reply_error(out, "ERR bad");
    reply_int(out, -42);
    reply_bulk(out, "hi");
    reply_null(out);
    reply_array_header(out, 2);
    EXPECT_EQ(out, "+OK\r\n-ERR bad\r\n:-42\r\n$2\r\nhi\r\n$-1\r\n*2\r\n");
}

TEST(Resp, ParseInt64) {
    int64_t v;
    EXPECT_TRUE(parse_int64("9223372036854775807", v));
    EXPECT_EQ(v, INT64_MAX);
    EXPECT_TRUE(parse_int64("-9223372036854775808", v));
    EXPECT_FALSE(parse_int64("9223372036854775808", v));
    EXPECT_FALSE(parse_int64("12a", v));
    EXPECT_FALSE(parse_int64(" 1", v));
    EXPECT_FALSE(parse_int64("", v));
    EXPECT_FALSE(parse_int64("+1", v));
}
