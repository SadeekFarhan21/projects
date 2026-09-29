#include "server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "resp.h"

namespace kv {

namespace {

#if defined(__linux__)
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;  // SO_NOSIGPIPE is set per socket instead
#endif

constexpr size_t kReadChunk = 16 * 1024;

int64_t wall_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int64_t mono_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

double cpu_seconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6;
}

bool set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

void tune_client_socket(int fd) {
    int one = 1;
    // Replies are small and latency matters; do not let Nagle hold them back.
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

}  // namespace

Server::Server(ServerConfig cfg) : cfg_(std::move(cfg)), proc_(store_) {}

Server::~Server() {
    for (auto& [fd, c] : clients_) ::close(fd);
    if (listen_fd_ >= 0) ::close(listen_fd_);
    if (aof_) aof_->close();
}

bool Server::start(std::string* err) {
    if (cfg_.aof_enabled) {
        // Replay through a processor with no propagation hook, so replayed
        // commands are not appended to the file again.
        CommandProcessor replayer(store_);
        std::string scratch;
        int64_t now = wall_ms();
        int64_t t0_wall = mono_ms();
        double t0_cpu = cpu_seconds();
        auto st = Aof::replay(cfg_.aof_path, [&](const std::vector<std::string>& a) {
            scratch.clear();
            replayer.execute(a, now, scratch);
        });
        if (!st.ok) {
            if (err) *err = st.error;
            return false;
        }
        size_t purged = store_.purge_expired(wall_ms());
        stats_.aof_replayed = st.commands;
        stats_.aof_truncated_bytes = st.truncated_bytes;
        // Report CPU time as well as wall time: on a shared, loaded machine
        // wall time mostly measures the scheduler.
        if (st.commands || st.truncated_bytes)
            std::fprintf(stderr,
                         "AOF: replayed %llu commands (%llu bytes), truncated %llu torn bytes, "
                         "purged %zu expired, %zu keys, %lld ms wall, %.0f ms cpu\n",
                         (unsigned long long)st.commands, (unsigned long long)st.bytes,
                         (unsigned long long)st.truncated_bytes, purged, store_.size(),
                         (long long)(mono_ms() - t0_wall), (cpu_seconds() - t0_cpu) * 1000.0);
        aof_ = std::make_unique<Aof>(cfg_.aof_path, cfg_.fsync);
        if (!aof_->open(err)) return false;
        Aof* aof = aof_.get();
        proc_.set_propagate([aof](const std::vector<std::string>& argv) { aof->append(argv); });
    }

    poller_ = make_poller();
    if (!poller_) {
        if (err) *err = "failed to create poller";
        return false;
    }

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        if (err) *err = std::string("socket: ") + std::strerror(errno);
        return false;
    }
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(cfg_.port));
    if (inet_pton(AF_INET, cfg_.bind.c_str(), &addr.sin_addr) != 1) {
        if (err) *err = "bad bind address " + cfg_.bind;
        return false;
    }
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(listen_fd_, 511) != 0 || !set_nonblocking(listen_fd_)) {
        if (err) *err = std::string("bind/listen: ") + std::strerror(errno);
        return false;
    }
    socklen_t len = sizeof(addr);
    getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    poller_->add(listen_fd_);
    if (cfg_.verbose)
        std::fprintf(stderr, "listening on %s:%d (%s)\n", cfg_.bind.c_str(), port_, poller_->name());
    return true;
}

void Server::accept_clients() {
    for (;;) {
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) continue;
            return;  // EAGAIN: backlog drained (or EMFILE: try again next event)
        }
        if (!set_nonblocking(fd) || !poller_->add(fd)) {
            ::close(fd);
            continue;
        }
        tune_client_socket(fd);
        auto c = std::make_unique<Client>();
        c->fd = fd;
        clients_[fd] = std::move(c);
        ++stats_.connections_accepted;
    }
}

void Server::close_client(Client& c) {
    int fd = c.fd;
    poller_->remove(fd);
    ::close(fd);
    clients_.erase(fd);  // destroys c
}

void Server::mark_pending(Client& c) {
    if (!c.pending) {
        c.pending = true;
        pending_.push_back(c.fd);
    }
}

bool Server::on_readable(Client& c) {
    if (c.read_paused) return true;  // stale event from before the pause
    size_t old = c.in.size();
    c.in.resize(old + kReadChunk);
    ssize_t n = ::read(c.fd, c.in.data() + old, kReadChunk);
    if (n <= 0) {
        c.in.resize(old);
        if (n < 0 && (errno == EAGAIN || errno == EINTR)) return true;
        close_client(c);  // EOF or hard error
        return false;
    }
    c.in.resize(old + static_cast<size_t>(n));
    if (c.in.size() - c.in_pos > cfg_.max_query_buf) {
        std::fprintf(stderr, "closing client fd=%d: query buffer over limit\n", c.fd);
        close_client(c);
        return false;
    }
    process_input(c);
    return true;
}

// Parse and execute as many complete commands as the buffer holds. This loop
// is what makes pipelining work: 100 commands that arrive in one read() are
// executed back to back and their replies coalesce into one write().
void Server::process_input(Client& c) {
    int64_t now = wall_ms();  // one clock read per batch, like Redis's cached mstime
    while (!c.close_after_write && c.in_pos < c.in.size()) {
        if (c.out.size() - c.out_pos > cfg_.out_soft_limit) {
            // Backpressure: the client is sending faster than it reads. Stop
            // reading until its replies drain; try_write() resumes us.
            c.read_paused = true;
            poller_->set_read(c.fd, false);
            ++stats_.read_pauses;
            break;
        }
        ParseResult r = parse_request(std::string_view(c.in).substr(c.in_pos), args_);
        if (r.status == ParseStatus::Incomplete) break;
        if (r.status == ParseStatus::Error) {
            reply_error(c.out, "ERR " + r.error);
            c.close_after_write = true;
            c.in.clear();
            c.in_pos = 0;
            break;
        }
        c.in_pos += r.consumed;
        if (args_.empty()) continue;
        if (proc_.execute(args_, now, c.out) == CommandProcessor::Action::Close) c.close_after_write = true;
    }
    // Compact the input buffer: reset when fully consumed, otherwise drop the
    // consumed prefix once it dominates, so memmove cost stays amortized O(1).
    if (c.in_pos == c.in.size()) {
        c.in.clear();
        c.in_pos = 0;
    } else if (c.in_pos > 64 * 1024 && c.in_pos * 2 > c.in.size()) {
        c.in.erase(0, c.in_pos);
        c.in_pos = 0;
    }
    if (c.out.size() > c.out_pos || c.close_after_write) mark_pending(c);
}

bool Server::try_write(Client& c) {
    for (;;) {
        while (c.out_pos < c.out.size()) {
            ssize_t n = ::send(c.fd, c.out.data() + c.out_pos, c.out.size() - c.out_pos, kSendFlags);
            if (n > 0) {
                c.out_pos += static_cast<size_t>(n);
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && errno == EAGAIN) {
                // Socket buffer full: wait for writability. Drop the sent
                // prefix if it is large so the buffer does not grow forever.
                if (c.out_pos > (1 << 20)) {
                    c.out.erase(0, c.out_pos);
                    c.out_pos = 0;
                }
                poller_->set_write(c.fd, true);
                return true;
            }
            close_client(c);  // EPIPE, ECONNRESET, ...
            return false;
        }
        c.out.clear();
        c.out_pos = 0;
        poller_->set_write(c.fd, false);
        if (c.close_after_write) {
            close_client(c);
            return false;
        }
        if (!c.read_paused) return true;
        // Output drained: resume a client paused by backpressure. Its input
        // buffer may already hold complete commands that no new read event
        // will ever announce, so process them now rather than waiting.
        c.read_paused = false;
        poller_->set_read(c.fd, true);
        process_input(c);
        if (c.out.empty()) return true;
    }
}

void Server::before_sleep() {
    if (aof_ && !aof_->flush() && !aof_error_logged_) {
        std::fprintf(stderr, "AOF write failed: %s (will retry)\n", std::strerror(errno));
        aof_error_logged_ = true;
    }
    std::vector<int> batch;
    batch.swap(pending_);
    for (int fd : batch) {
        auto it = clients_.find(fd);
        if (it == clients_.end()) continue;
        Client& c = *it->second;
        c.pending = false;
        try_write(c);
    }
}

void Server::cron(int64_t now_mono) {
    int64_t period_us = 1000000 / std::max(1, cfg_.hz);
    auto st = store_.active_expire_cycle(wall_ms(), period_us / 4);  // at most 25% of a tick
    stats_.expired_active += st.expired;
    ++stats_.expire_cycles;
    if (aof_) aof_->tick(now_mono);
}

void Server::run() {
    const int64_t period = 1000 / std::max(1, cfg_.hz);
    int64_t next_cron = mono_ms() + period;
    std::vector<PollEvent> events;
    while (!stop_.load(std::memory_order_relaxed)) {
        before_sleep();
        int64_t now = mono_ms();
        int timeout = pending_.empty() ? static_cast<int>(std::max<int64_t>(0, next_cron - now)) : 0;
        poller_->wait(events, timeout);
        for (const PollEvent& ev : events) {
            if (ev.fd == listen_fd_) {
                accept_clients();
                continue;
            }
            auto it = clients_.find(ev.fd);
            if (it == clients_.end()) continue;  // closed earlier in this batch
            Client& c = *it->second;
            if (ev.writable && !try_write(c)) continue;
            if (ev.readable) on_readable(c);
        }
        now = mono_ms();
        if (now >= next_cron) {
            cron(now);
            next_cron = now + period;
        }
    }
    before_sleep();  // flush AOF and any last replies
    if (aof_) aof_->close();
    for (auto& [fd, c] : clients_) ::close(fd);
    clients_.clear();
    ::close(listen_fd_);
    listen_fd_ = -1;
}

ServerStats Server::stats() const {
    ServerStats s = stats_;
    s.commands = proc_.commands_processed();
    return s;
}

}  // namespace kv
