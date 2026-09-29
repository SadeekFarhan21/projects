// kvbench: closed-loop, pipelined load generator for any RESP server.
//
// C connections are spread over T threads. Each connection keeps up to P
// requests in flight: it starts by sending P, and every time replies arrive it
// tops the window back up to P. Latency of a request is measured from the
// moment its batch was written to the moment its reply was parsed, so at
// P > 1 it includes the time spent queued behind earlier requests in the same
// pipeline. That is the latency a pipelining client actually observes.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string host = "127.0.0.1";
    int port = 6379;
    int clients = 50;
    int pipeline = 1;
    long long requests = 1000000;
    int threads = 0;  // 0: min(clients, 8)
    int value_size = 16;
    long long keyspace = 100000;
    std::string workload = "set";  // set | get | mix | ping
    std::string label;
    bool json = false;
};

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// Returns the number of bytes of one complete RESP reply at buf[pos..], 0 if
// incomplete. Sets is_error for '-' replies.
size_t reply_len(const std::string& buf, size_t pos, bool& is_error) {
    if (pos >= buf.size()) return 0;
    size_t eol = buf.find("\r\n", pos);
    if (eol == std::string::npos) return 0;
    char t = buf[pos];
    if (t == '-') is_error = true;
    if (t == '+' || t == '-' || t == ':') return eol + 2 - pos;
    long long n = std::strtoll(buf.c_str() + pos + 1, nullptr, 10);
    if (t == '$') {
        if (n < 0) return eol + 2 - pos;
        size_t end = eol + 2 + static_cast<size_t>(n) + 2;
        return end <= buf.size() ? end - pos : 0;
    }
    if (t == '*') {
        size_t p = eol + 2;
        for (long long i = 0; i < n; ++i) {
            size_t l = reply_len(buf, p, is_error);
            if (l == 0) return 0;
            p += l;
        }
        return p - pos;
    }
    std::fprintf(stderr, "unexpected reply byte '%c'\n", t);
    std::exit(1);
}

struct Conn {
    int fd = -1;
    long long quota = 0;     // requests this connection must complete
    long long sent = 0;
    long long done = 0;
    std::deque<uint64_t> inflight;  // send timestamps, FIFO matches reply order
    std::string rbuf;
    size_t rpos = 0;
};

struct ThreadResult {
    std::vector<uint32_t> lat_ns;  // capped at ~4.29 s, plenty for this use
    long long errors = 0;
};

int connect_to(const Options& o) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(o.port));
    inet_pton(AF_INET, o.host.c_str(), &a.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
        std::perror("connect");
        std::exit(1);
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    return fd;
}

void append_bulk(std::string& out, const char* s, size_t n) {
    out += '$';
    out += std::to_string(n);
    out += "\r\n";
    out.append(s, n);
    out += "\r\n";
}

struct Rng {
    uint64_t s;
    uint64_t next() {  // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
};

void build_request(std::string& out, const Options& o, Rng& rng, const std::string& value) {
    char key[32];
    int kl = std::snprintf(key, sizeof(key), "key:%012llu",
                           static_cast<unsigned long long>(rng.next() % static_cast<uint64_t>(o.keyspace)));
    bool is_set = o.workload == "set" || (o.workload == "mix" && (rng.next() & 1));
    if (o.workload == "ping") {
        out += "*1\r\n$4\r\nPING\r\n";
    } else if (is_set) {
        out += "*3\r\n$3\r\nSET\r\n";
        append_bulk(out, key, static_cast<size_t>(kl));
        append_bulk(out, value.data(), value.size());
    } else {
        out += "*2\r\n$3\r\nGET\r\n";
        append_bulk(out, key, static_cast<size_t>(kl));
    }
}

void send_all(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::send(fd, s.data() + off, s.size() - off, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::perror("send");
            std::exit(1);
        }
        off += static_cast<size_t>(n);
    }
}

void top_up(Conn& c, const Options& o, Rng& rng, const std::string& value, std::string& scratch) {
    long long room = o.pipeline - static_cast<long long>(c.inflight.size());
    long long k = std::min(room, c.quota - c.sent);
    if (k <= 0) return;
    scratch.clear();
    for (long long i = 0; i < k; ++i) build_request(scratch, o, rng, value);
    uint64_t t = now_ns();
    send_all(c.fd, scratch);
    for (long long i = 0; i < k; ++i) c.inflight.push_back(t);
    c.sent += k;
}

void run_thread(const Options& o, std::vector<Conn>& conns, ThreadResult& res, uint64_t seed) {
    Rng rng{seed * 0x9E3779B97F4A7C15ULL + 1};
    std::string value(static_cast<size_t>(o.value_size), 'x');
    std::string scratch;
    long long total = 0;
    for (auto& c : conns) total += c.quota;
    res.lat_ns.reserve(static_cast<size_t>(total));

    for (auto& c : conns) top_up(c, o, rng, value, scratch);
    std::vector<pollfd> pfds(conns.size());
    char buf[64 * 1024];
    long long remaining = total;
    while (remaining > 0) {
        for (size_t i = 0; i < conns.size(); ++i) pfds[i] = {conns[i].fd, POLLIN, 0};
        if (::poll(pfds.data(), pfds.size(), 5000) <= 0) {
            std::fprintf(stderr, "poll timeout/error with %lld requests outstanding\n", remaining);
            std::exit(1);
        }
        for (size_t i = 0; i < conns.size(); ++i) {
            if (!(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            Conn& c = conns[i];
            ssize_t n = ::recv(c.fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                std::fprintf(stderr, "server closed connection\n");
                std::exit(1);
            }
            c.rbuf.append(buf, static_cast<size_t>(n));
            uint64_t t = now_ns();
            for (;;) {
                bool err = false;
                size_t l = reply_len(c.rbuf, c.rpos, err);
                if (l == 0) break;
                c.rpos += l;
                uint64_t d = t - c.inflight.front();
                c.inflight.pop_front();
                res.lat_ns.push_back(static_cast<uint32_t>(std::min<uint64_t>(d, UINT32_MAX)));
                res.errors += err;
                ++c.done;
                --remaining;
            }
            if (c.rpos == c.rbuf.size()) {
                c.rbuf.clear();
                c.rpos = 0;
            } else if (c.rpos > 64 * 1024) {
                c.rbuf.erase(0, c.rpos);
                c.rpos = 0;
            }
            top_up(c, o, rng, value, scratch);
        }
    }
}

double pct(const std::vector<uint32_t>& sorted, double p) {
    if (sorted.empty()) return 0;
    size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(sorted.size() - 1) + 0.5);
    return sorted[std::min(idx, sorted.size() - 1)] / 1000.0;  // microseconds
}

void usage() {
    std::fprintf(stderr,
                 "usage: kvbench [-h host] [-p port] [-c clients] [-P pipeline] [-n requests]\n"
                 "               [-t threads] [-d value_bytes] [-r keyspace] [-w set|get|mix|ping]\n"
                 "               [--label L] [--json]\n");
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h") o.host = next();
        else if (a == "-p") o.port = std::atoi(next());
        else if (a == "-c") o.clients = std::atoi(next());
        else if (a == "-P") o.pipeline = std::atoi(next());
        else if (a == "-n") o.requests = std::atoll(next());
        else if (a == "-t") o.threads = std::atoi(next());
        else if (a == "-d") o.value_size = std::atoi(next());
        else if (a == "-r") o.keyspace = std::atoll(next());
        else if (a == "-w") o.workload = next();
        else if (a == "--label") o.label = next();
        else if (a == "--json") o.json = true;
        else {
            usage();
            return 2;
        }
    }
    if (o.clients < 1 || o.pipeline < 1 || o.requests < o.clients) {
        usage();
        return 2;
    }
    int T = o.threads > 0 ? o.threads : std::min(o.clients, 8);
    T = std::min(T, o.clients);

    // Distribute connections round-robin over threads and requests evenly over connections.
    std::vector<std::vector<Conn>> per_thread(static_cast<size_t>(T));
    for (int i = 0; i < o.clients; ++i) {
        Conn c;
        c.fd = connect_to(o);
        c.quota = o.requests / o.clients + (i < o.requests % o.clients ? 1 : 0);
        per_thread[static_cast<size_t>(i % T)].push_back(std::move(c));
    }
    std::vector<ThreadResult> results(static_cast<size_t>(T));

    auto t0 = Clock::now();
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t)
        ths.emplace_back(run_thread, std::cref(o), std::ref(per_thread[static_cast<size_t>(t)]),
                         std::ref(results[static_cast<size_t>(t)]), static_cast<uint64_t>(t + 1));
    for (auto& th : ths) th.join();
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();

    std::vector<uint32_t> all;
    long long errors = 0;
    for (auto& r : results) {
        all.insert(all.end(), r.lat_ns.begin(), r.lat_ns.end());
        errors += r.errors;
    }
    std::sort(all.begin(), all.end());
    for (auto& v : per_thread)
        for (auto& c : v) ::close(c.fd);

    double ops = static_cast<double>(all.size()) / secs;
    if (o.json) {
        std::printf(
            "{\"label\":\"%s\",\"workload\":\"%s\",\"clients\":%d,\"pipeline\":%d,\"threads\":%d,"
            "\"requests\":%zu,\"value_size\":%d,\"seconds\":%.4f,\"ops_per_sec\":%.0f,"
            "\"p50_us\":%.1f,\"p90_us\":%.1f,\"p99_us\":%.1f,\"p999_us\":%.1f,\"max_us\":%.1f,\"errors\":%lld}\n",
            o.label.c_str(), o.workload.c_str(), o.clients, o.pipeline, T, all.size(), o.value_size, secs, ops,
            pct(all, 50), pct(all, 90), pct(all, 99), pct(all, 99.9), pct(all, 100), errors);
    } else {
        std::printf("%s: %zu requests, %d clients, pipeline %d, %d threads, %d byte values\n", o.workload.c_str(),
                    all.size(), o.clients, o.pipeline, T, o.value_size);
        std::printf("  %.2f s, %.0f ops/sec, errors %lld\n", secs, ops, errors);
        std::printf("  latency us: p50 %.1f  p90 %.1f  p99 %.1f  p99.9 %.1f  max %.1f\n", pct(all, 50),
                    pct(all, 90), pct(all, 99), pct(all, 99.9), pct(all, 100));
    }
    return errors > 0 ? 3 : 0;
}
