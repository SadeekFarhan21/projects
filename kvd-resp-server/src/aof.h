// Append-only file persistence.
//
// Every successful write command is appended to an in-memory buffer as a RESP
// array (the exact same bytes a client would send). The event loop calls
// flush() once per iteration, *before* it writes any replies to sockets, so a
// client never sees an OK for a write that has not at least reached the
// kernel. fsync policy then decides when the kernel is forced to hit disk:
//
//   always    fsync on every flush that wrote bytes (before replies go out)
//   everysec  fsync from the 100 ms cron if >= 1 s has passed since the last one
//   no        never fsync; leave it to the OS
//
// On startup the file is replayed through the normal command path. A torn
// final record (crash mid-write) is truncated away; corruption anywhere else
// aborts startup, since silently dropping data in the middle is worse.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kv {

enum class FsyncPolicy { Always, EverySec, No };

bool parse_fsync_policy(const std::string& s, FsyncPolicy& out);

class Aof {
public:
    Aof(std::string path, FsyncPolicy policy);
    ~Aof();
    Aof(const Aof&) = delete;
    Aof& operator=(const Aof&) = delete;

    bool open(std::string* err);  // open for append, creating if needed
    void append(const std::vector<std::string>& argv);
    // Write the buffer to the file. Returns false on a write error.
    bool flush();
    // Called from the cron; performs the everysec fsync. now_ms is monotonic.
    void tick(int64_t now_ms);
    // Final flush + fsync on shutdown.
    void close();

    size_t pending_bytes() const { return buf_.size(); }
    uint64_t fsync_count() const { return fsyncs_; }
    const std::string& path() const { return path_; }

    struct ReplayStats {
        uint64_t commands = 0;
        uint64_t bytes = 0;
        uint64_t truncated_bytes = 0;  // torn tail removed
        bool ok = true;
        std::string error;
    };
    // Replay `path` (missing file is fine: zero commands). Calls `apply` for
    // each command. If the tail is an incomplete record, the file is
    // truncated to the last complete record.
    static ReplayStats replay(const std::string& path,
                              const std::function<void(const std::vector<std::string>&)>& apply);

private:
    std::string path_;
    FsyncPolicy policy_;
    int fd_ = -1;
    std::string buf_;
    bool dirty_since_fsync_ = false;
    int64_t last_fsync_ms_ = 0;
    uint64_t fsyncs_ = 0;
};

}  // namespace kv
