// Single-threaded event-loop TCP server.
//
// One thread owns everything: the listening socket, every client connection,
// the keyspace and the AOF. Each loop iteration is
//
//   before_sleep()   flush AOF buffer (and fsync if appendfsync=always),
//                    then write pending replies to sockets
//   poller.wait()    block until a socket is ready or the next cron tick
//   dispatch events  accept / read + parse + execute (pipelined) / write
//   cron()           every 1000/hz ms: active expiry cycle, AOF everysec fsync
//
// Because there is no concurrency, commands are trivially atomic and no locks
// are needed anywhere.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "aof.h"
#include "commands.h"
#include "poller.h"
#include "store.h"

namespace kv {

struct ServerConfig {
    std::string bind = "127.0.0.1";
    int port = 6379;             // 0 picks an ephemeral port (tests)
    bool aof_enabled = true;
    std::string aof_path = "appendonly.aof";
    FsyncPolicy fsync = FsyncPolicy::EverySec;
    int hz = 10;                 // cron frequency
    size_t out_soft_limit = 1 << 20;  // stop reading a client whose unsent replies exceed this
    size_t max_query_buf = 1ULL << 30;  // close a client whose unparsed input exceeds this
    bool verbose = false;
};

struct ServerStats {
    uint64_t connections_accepted = 0;
    uint64_t commands = 0;
    uint64_t expired_active = 0;
    uint64_t expire_cycles = 0;
    uint64_t read_pauses = 0;   // times backpressure paused a client's reads
    uint64_t aof_replayed = 0;
    uint64_t aof_truncated_bytes = 0;
};

class Server {
public:
    explicit Server(ServerConfig cfg);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Replay the AOF, open it for append, bind and listen. Returns false with
    // a message on failure.
    bool start(std::string* err);
    // Run the event loop until request_stop(). Flushes and fsyncs the AOF on exit.
    void run();
    // Safe to call from another thread or a signal handler.
    void request_stop() noexcept { stop_.store(true, std::memory_order_relaxed); }

    int port() const { return port_; }
    Store& store() { return store_; }
    ServerStats stats() const;
    size_t client_count() const { return clients_.size(); }

private:
    struct Client {
        int fd = -1;
        std::string in;
        size_t in_pos = 0;       // bytes of `in` already consumed by the parser
        std::string out;
        size_t out_pos = 0;      // bytes of `out` already written to the socket
        bool close_after_write = false;
        bool read_paused = false;
        bool pending = false;    // in pending_ list
    };

    void accept_clients();
    bool on_readable(Client& c);       // returns false if the client was closed
    void process_input(Client& c);
    bool try_write(Client& c);         // returns false if the client was closed
    void close_client(Client& c);
    void mark_pending(Client& c);
    void before_sleep();
    void cron(int64_t mono_ms);

    ServerConfig cfg_;
    Store store_;
    CommandProcessor proc_;
    std::unique_ptr<Aof> aof_;
    std::unique_ptr<Poller> poller_;
    int listen_fd_ = -1;
    int port_ = 0;
    std::atomic<bool> stop_{false};
    std::unordered_map<int, std::unique_ptr<Client>> clients_;
    std::vector<int> pending_;         // clients with unsent output
    std::vector<std::string> args_;    // reused parse buffer
    ServerStats stats_;
    bool aof_error_logged_ = false;
};

}  // namespace kv
