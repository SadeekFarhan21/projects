#include "aof.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "resp.h"

namespace kv {

bool parse_fsync_policy(const std::string& s, FsyncPolicy& out) {
    if (s == "always") out = FsyncPolicy::Always;
    else if (s == "everysec") out = FsyncPolicy::EverySec;
    else if (s == "no") out = FsyncPolicy::No;
    else return false;
    return true;
}

Aof::Aof(std::string path, FsyncPolicy policy) : path_(std::move(path)), policy_(policy) {}

Aof::~Aof() { close(); }

bool Aof::open(std::string* err) {
    fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd_ < 0) {
        if (err) *err = "open " + path_ + ": " + std::strerror(errno);
        return false;
    }
    return true;
}

void Aof::append(const std::vector<std::string>& argv) { encode_command(buf_, argv); }

bool Aof::flush() {
    if (fd_ < 0 || buf_.empty()) return true;
    size_t off = 0;
    while (off < buf_.size()) {
        ssize_t n = ::write(fd_, buf_.data() + off, buf_.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            // Keep the unwritten part so the next flush retries it.
            buf_.erase(0, off);
            return false;
        }
        off += static_cast<size_t>(n);
    }
    buf_.clear();
    dirty_since_fsync_ = true;
    if (policy_ == FsyncPolicy::Always) {
        // Note: on macOS fsync() pushes data to the drive but not through the
        // drive's write cache; F_FULLFSYNC would. We use plain fsync, matching
        // what Redis does on Linux, and document the difference.
        ::fsync(fd_);
        ++fsyncs_;
        dirty_since_fsync_ = false;
    }
    return true;
}

void Aof::tick(int64_t now_ms) {
    if (fd_ < 0 || policy_ != FsyncPolicy::EverySec) return;
    if (dirty_since_fsync_ && now_ms - last_fsync_ms_ >= 1000) {
        ::fsync(fd_);
        ++fsyncs_;
        dirty_since_fsync_ = false;
        last_fsync_ms_ = now_ms;
    }
}

void Aof::close() {
    if (fd_ < 0) return;
    flush();
    ::fsync(fd_);
    ::close(fd_);
    fd_ = -1;
}

Aof::ReplayStats Aof::replay(const std::string& path,
                             const std::function<void(const std::vector<std::string>&)>& apply) {
    ReplayStats st;
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (errno != ENOENT) {
            st.ok = false;
            st.error = "open " + path + ": " + std::strerror(errno);
        }
        return st;
    }
    // Read the whole file. Fine for v0; a streaming reader would bound memory.
    std::string data;
    char chunk[1 << 16];
    for (;;) {
        ssize_t n = ::read(fd, chunk, sizeof(chunk));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        data.append(chunk, static_cast<size_t>(n));
    }
    ::close(fd);

    std::vector<std::string> args;
    size_t pos = 0;
    while (pos < data.size()) {
        ParseResult r = parse_request(std::string_view(data).substr(pos), args);
        if (r.status == ParseStatus::Incomplete) break;
        if (r.status == ParseStatus::Error || data[pos] != '*') {
            st.ok = false;
            st.error = "corrupt AOF at byte " + std::to_string(pos) + ": " +
                       (r.error.empty() ? "not a RESP array" : r.error);
            return st;
        }
        if (!args.empty()) {
            apply(args);
            ++st.commands;
        }
        pos += r.consumed;
    }
    st.bytes = pos;
    if (pos < data.size()) {
        // Torn write at the tail: drop it so future appends start on a record
        // boundary. Real Redis does the same with aof-load-truncated yes.
        st.truncated_bytes = data.size() - pos;
        if (::truncate(path.c_str(), static_cast<off_t>(pos)) != 0) {
            st.ok = false;
            st.error = "truncate " + path + ": " + std::strerror(errno);
        }
    }
    return st;
}

}  // namespace kv
