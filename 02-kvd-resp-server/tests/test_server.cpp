// End-to-end tests: a real Server on an ephemeral port in a background thread,
// talked to over real TCP sockets.
#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <thread>

#include "resp.h"
#include "server.h"

using namespace kv;
using namespace std::chrono_literals;

namespace {

class RunningServer {
public:
    explicit RunningServer(ServerConfig cfg) : server_(std::move(cfg)) {
        std::string err;
        if (!server_.start(&err)) throw std::runtime_error(err);
        thread_ = std::thread([this] { server_.run(); });
    }
    ~RunningServer() { stop(); }
    void stop() {
        if (thread_.joinable()) {
            server_.request_stop();
            thread_.join();
        }
    }
    int port() const { return server_.port(); }
    Server& server() { return server_; }

private:
    Server server_;
    std::thread thread_;
};

ServerConfig test_config(const std::string& aof_path = "") {
    ServerConfig c;
    c.port = 0;
    c.aof_enabled = !aof_path.empty();
    c.aof_path = aof_path;
    return c;
}

// Minimal blocking RESP client.
class Client {
public:
    explicit Client(int port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(static_cast<uint16_t>(port));
        inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) throw std::runtime_error("connect");
        int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
        setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
        timeval tv{10, 0};
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }
    ~Client() { ::close(fd_); }
    int fd() const { return fd_; }

    void send_raw(std::string_view s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t n = ::send(fd_, s.data() + off, s.size() - off, 0);
            if (n <= 0) throw std::runtime_error("send failed");
            off += static_cast<size_t>(n);
        }
    }
    void send_cmd(const std::vector<std::string>& a) {
        std::string s;
        encode_command(s, a);
        send_raw(s);
    }
    // Read exactly one reply and return its raw bytes. Empty string on EOF.
    std::string read_reply() {
        for (;;) {
            size_t len = reply_len(buf_, 0);
            if (len) {
                std::string r = buf_.substr(0, len);
                buf_.erase(0, len);
                return r;
            }
            char tmp[65536];
            ssize_t n = ::recv(fd_, tmp, sizeof(tmp), 0);
            if (n <= 0) return {};
            buf_.append(tmp, static_cast<size_t>(n));
        }
    }
    std::string call(const std::vector<std::string>& a) {
        send_cmd(a);
        return read_reply();
    }

private:
    static size_t reply_len(const std::string& b, size_t pos) {
        size_t eol = b.find("\r\n", pos);
        if (eol == std::string::npos) return 0;
        char t = b[pos];
        if (t == '+' || t == '-' || t == ':') return eol + 2 - pos;
        long long n = std::stoll(b.substr(pos + 1, eol - pos - 1));
        if (t == '$') {
            if (n < 0) return eol + 2 - pos;
            size_t end = eol + 2 + static_cast<size_t>(n) + 2;
            return end <= b.size() ? end - pos : 0;
        }
        size_t p = eol + 2;  // '*'
        for (long long i = 0; i < n; ++i) {
            size_t l = reply_len(b, p);
            if (!l) return 0;
            p += l;
        }
        return p - pos;
    }
    int fd_;
    std::string buf_;
};

std::string temp_aof(const char* tag) {
    std::string p = ::testing::TempDir() + "kv_server_" + tag + "_" + std::to_string(::getpid()) + ".aof";
    ::unlink(p.c_str());
    return p;
}

}  // namespace

TEST(Server, BasicCommandsOverTcp) {
    RunningServer s(test_config());
    Client c(s.port());
    EXPECT_EQ(c.call({"PING"}), "+PONG\r\n");
    EXPECT_EQ(c.call({"SET", "k", "v", "EX", "100"}), "+OK\r\n");
    EXPECT_EQ(c.call({"GET", "k"}), "$1\r\nv\r\n");
    EXPECT_EQ(c.call({"TTL", "k"}), ":100\r\n");
    EXPECT_EQ(c.call({"INCR", "n"}), ":1\r\n");
    EXPECT_EQ(c.call({"KEYS", "k"}), "*1\r\n$1\r\nk\r\n");
    EXPECT_EQ(c.call({"DEL", "k", "n"}), ":2\r\n");
    EXPECT_EQ(c.call({"EXISTS", "k"}), ":0\r\n");
}

TEST(Server, InlineCommandsWork) {
    RunningServer s(test_config());
    Client c(s.port());
    c.send_raw("SET a 1\r\nGET a\r\n");
    EXPECT_EQ(c.read_reply(), "+OK\r\n");
    EXPECT_EQ(c.read_reply(), "$1\r\n1\r\n");
}

TEST(Server, DeepPipelineRepliesInOrder) {
    RunningServer s(test_config());
    Client c(s.port());
    constexpr int N = 20000;
    std::string batch;
    for (int i = 0; i < N; ++i) encode_command(batch, {"INCR", "ctr"});
    c.send_raw(batch);  // one write, 20k commands
    for (int i = 1; i <= N; ++i) ASSERT_EQ(c.read_reply(), ":" + std::to_string(i) + "\r\n");
}

TEST(Server, CommandSplitAcrossManyPackets) {
    RunningServer s(test_config());
    Client c(s.port());
    std::string cmd;
    encode_command(cmd, {"SET", "split", "value-sent-one-byte-at-a-time"});
    for (char ch : cmd) {
        c.send_raw(std::string_view(&ch, 1));
        std::this_thread::sleep_for(100us);
    }
    EXPECT_EQ(c.read_reply(), "+OK\r\n");
    EXPECT_EQ(c.call({"GET", "split"}), "$29\r\nvalue-sent-one-byte-at-a-time\r\n");
}

TEST(Server, LargeValueRoundTrip) {
    RunningServer s(test_config());
    Client c(s.port());
    std::string big(8 << 20, 'z');  // 8 MB: forces partial writes on both sides
    big[12345] = 'a';
    EXPECT_EQ(c.call({"SET", "big", big}), "+OK\r\n");
    std::string r = c.call({"GET", "big"});
    ASSERT_EQ(r.size(), big.size() + 12);  // "$8388608\r\n" + payload + "\r\n"
    EXPECT_EQ(r.substr(10, big.size()), big);
}

TEST(Server, BackpressurePausesAndResumesReads) {
    // A client that writes a large pipeline before reading anything. The
    // server must stop reading once unsent replies pass the soft limit, then
    // resume and finish every command once the client drains its replies.
    // A small soft limit makes the pause deterministic: it no longer depends
    // on how fast the kernel drains socket buffers (which differs a lot
    // between the -O2 and the ASan build).
    ServerConfig cfg = test_config();
    cfg.out_soft_limit = 64 * 1024;
    RunningServer s(cfg);
    Client c(s.port());
    EXPECT_EQ(c.call({"SET", "v", std::string(1024, 'x')}), "+OK\r\n");
    constexpr int N = 30000;  // ~30 MB of replies
    std::thread writer([&] {
        std::string batch;
        for (int i = 0; i < N; ++i) encode_command(batch, {"GET", "v"});
        c.send_raw(batch);
    });
    std::this_thread::sleep_for(200ms);  // let buffers fill so the pause really happens
    std::string want = "$1024\r\n" + std::string(1024, 'x') + "\r\n";
    for (int i = 0; i < N; ++i) ASSERT_EQ(c.read_reply(), want) << "reply " << i;
    writer.join();
    EXPECT_EQ(c.call({"PING"}), "+PONG\r\n");
    s.stop();
    EXPECT_GT(s.server().stats().read_pauses, 0u);
}

TEST(Server, ManyConcurrentClients) {
    RunningServer s(test_config());
    constexpr int kClients = 16, kIncrs = 2000;
    std::vector<std::thread> ts;
    for (int t = 0; t < kClients; ++t)
        ts.emplace_back([&] {
            Client c(s.port());
            for (int i = 0; i < kIncrs; ++i) c.call({"INCR", "shared"});
        });
    for (auto& t : ts) t.join();
    Client c(s.port());
    EXPECT_EQ(c.call({"GET", "shared"}), "$5\r\n32000\r\n");
}

TEST(Server, ProtocolErrorClosesConnection) {
    RunningServer s(test_config());
    Client c(s.port());
    c.send_raw("*1\r\n+oops\r\n");
    std::string r = c.read_reply();
    EXPECT_EQ(r.rfind("-ERR Protocol error", 0), 0u) << r;
    EXPECT_EQ(c.read_reply(), "");  // then EOF
}

TEST(Server, QuitClosesAfterReply) {
    RunningServer s(test_config());
    Client c(s.port());
    c.send_raw("*1\r\n$4\r\nQUIT\r\n*1\r\n$4\r\nPING\r\n");  // PING after QUIT is ignored
    EXPECT_EQ(c.read_reply(), "+OK\r\n");
    EXPECT_EQ(c.read_reply(), "");
}

TEST(Server, ActiveExpiryReclaimsUntouchedKeys) {
    RunningServer s(test_config());
    Client c(s.port());
    std::string batch;
    for (int i = 0; i < 5000; ++i) encode_command(batch, {"SET", "t" + std::to_string(i), "v", "PX", "100"});
    c.send_raw(batch);
    for (int i = 0; i < 5000; ++i) ASSERT_EQ(c.read_reply(), "+OK\r\n");
    // Nobody touches the keys, so lazy expiry never fires; only the cron can
    // remove them. DBSIZE counts keys still physically present. Poll with a
    // deadline instead of a fixed sleep: under ASan on a loaded machine the
    // SETs alone can take longer than the TTL.
    auto deadline = std::chrono::steady_clock::now() + 5s;
    std::string n;
    while (std::chrono::steady_clock::now() < deadline) {
        n = c.call({"DBSIZE"});
        if (n == ":0\r\n") break;
        std::this_thread::sleep_for(50ms);
    }
    EXPECT_EQ(n, ":0\r\n");
    s.stop();
    EXPECT_EQ(s.server().stats().expired_active, 5000u);
}

TEST(Server, AofPersistsAcrossRestart) {
    std::string path = temp_aof("restart");
    {
        RunningServer s(test_config(path));
        Client c(s.port());
        EXPECT_EQ(c.call({"SET", "keep", "1"}), "+OK\r\n");
        EXPECT_EQ(c.call({"SET", "gone", "1"}), "+OK\r\n");
        EXPECT_EQ(c.call({"DEL", "gone"}), ":1\r\n");
        EXPECT_EQ(c.call({"INCRBY", "n", "41"}), ":41\r\n");
        EXPECT_EQ(c.call({"INCR", "n"}), ":42\r\n");
        EXPECT_EQ(c.call({"SET", "ttl", "x", "EX", "1000"}), "+OK\r\n");
        EXPECT_EQ(c.call({"SET", "short", "x", "PX", "100"}), "+OK\r\n");
    }  // graceful stop: flush + fsync
    std::this_thread::sleep_for(150ms);  // let "short" expire in wall-clock time
    RunningServer s2(test_config(path));
    Client c(s2.port());
    EXPECT_EQ(c.call({"GET", "keep"}), "$1\r\n1\r\n");
    EXPECT_EQ(c.call({"EXISTS", "gone"}), ":0\r\n");
    EXPECT_EQ(c.call({"GET", "n"}), "$2\r\n42\r\n");
    std::string ttl = c.call({"TTL", "ttl"});
    EXPECT_TRUE(ttl == ":1000\r\n" || ttl == ":999\r\n") << ttl;
    EXPECT_EQ(c.call({"GET", "short"}), "$-1\r\n");
    s2.stop();
    EXPECT_EQ(s2.server().stats().aof_replayed, 7u);  // read after join: no data race
    ::unlink(path.c_str());
}
