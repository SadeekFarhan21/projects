#include <gtest/gtest.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

#include "aof.h"
#include "resp.h"

using namespace kv;

namespace {

std::string temp_path(const char* tag) {
    std::string p = ::testing::TempDir() + "kv_aof_" + tag + "_" + std::to_string(::getpid()) + ".aof";
    ::unlink(p.c_str());
    return p;
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::vector<std::string>> replay_all(const std::string& p, Aof::ReplayStats* st = nullptr) {
    std::vector<std::vector<std::string>> got;
    auto s = Aof::replay(p, [&](const std::vector<std::string>& a) { got.push_back(a); });
    if (st) *st = s;
    return got;
}

}  // namespace

TEST(Aof, MissingFileReplaysNothing) {
    Aof::ReplayStats st;
    auto got = replay_all(temp_path("missing"), &st);
    EXPECT_TRUE(st.ok);
    EXPECT_TRUE(got.empty());
}

TEST(Aof, AppendFlushReplayRoundTrip) {
    std::string p = temp_path("roundtrip");
    {
        Aof aof(p, FsyncPolicy::Always);
        ASSERT_TRUE(aof.open(nullptr));
        aof.append({"SET", "k", "v"});
        aof.append({"SET", "bin", std::string("a\r\nb\0c", 6)});
        EXPECT_GT(aof.pending_bytes(), 0u);
        ASSERT_TRUE(aof.flush());
        EXPECT_EQ(aof.pending_bytes(), 0u);
        EXPECT_EQ(aof.fsync_count(), 1u);
        aof.append({"DEL", "k"});
    }  // destructor flushes the last record
    auto got = replay_all(p);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[1][2], std::string("a\r\nb\0c", 6));
    EXPECT_EQ(got[2], (std::vector<std::string>{"DEL", "k"}));
    ::unlink(p.c_str());
}

TEST(Aof, EverysecFsyncsOnlyFromTick) {
    std::string p = temp_path("everysec");
    Aof aof(p, FsyncPolicy::EverySec);
    ASSERT_TRUE(aof.open(nullptr));
    aof.append({"SET", "a", "1"});
    aof.flush();
    EXPECT_EQ(aof.fsync_count(), 0u);
    aof.tick(5000);
    EXPECT_EQ(aof.fsync_count(), 1u);
    aof.tick(5500);  // nothing new written
    EXPECT_EQ(aof.fsync_count(), 1u);
    aof.append({"SET", "a", "2"});
    aof.flush();
    aof.tick(5600);  // written, but < 1 s since the last fsync
    EXPECT_EQ(aof.fsync_count(), 1u);
    aof.tick(6000);
    EXPECT_EQ(aof.fsync_count(), 2u);
    ::unlink(p.c_str());
}

TEST(Aof, TornTailIsTruncated) {
    std::string p = temp_path("torn");
    std::string good;
    encode_command(good, {"SET", "a", "1"});
    encode_command(good, {"SET", "b", "2"});
    std::string torn;
    encode_command(torn, {"SET", "c", "3"});
    torn.resize(torn.size() - 4);  // crash in the middle of the last record
    {
        std::ofstream f(p, std::ios::binary);
        f << good << torn;
    }
    Aof::ReplayStats st;
    auto got = replay_all(p, &st);
    EXPECT_TRUE(st.ok);
    EXPECT_EQ(got.size(), 2u);
    EXPECT_EQ(st.truncated_bytes, torn.size());
    EXPECT_EQ(slurp(p), good);  // file now ends on a record boundary

    // Appending after recovery yields a clean file.
    {
        Aof aof(p, FsyncPolicy::No);
        ASSERT_TRUE(aof.open(nullptr));
        aof.append({"SET", "d", "4"});
    }
    EXPECT_EQ(replay_all(p).size(), 3u);
    ::unlink(p.c_str());
}

TEST(Aof, CorruptionInTheMiddleFails) {
    std::string p = temp_path("corrupt");
    std::string data;
    encode_command(data, {"SET", "a", "1"});
    data += "GARBAGE\r\n";
    encode_command(data, {"SET", "b", "2"});
    {
        std::ofstream f(p, std::ios::binary);
        f << data;
    }
    Aof::ReplayStats st;
    replay_all(p, &st);
    EXPECT_FALSE(st.ok);
    EXPECT_NE(st.error.find("corrupt AOF at byte"), std::string::npos);
    ::unlink(p.c_str());
}
