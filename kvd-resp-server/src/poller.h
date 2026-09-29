// Readiness notification behind a small interface: kqueue on macOS/BSD,
// epoll on Linux. Both are used level-triggered, so a handler that does not
// drain a socket fully simply gets called again on the next wait().
#pragma once

#include <memory>
#include <vector>

namespace kv {

struct PollEvent {
    int fd = -1;
    bool readable = false;
    bool writable = false;
    bool error = false;  // EOF / error condition reported by the kernel
};

class Poller {
public:
    virtual ~Poller() = default;
    // Register fd with read interest on (and write interest off).
    virtual bool add(int fd) = 0;
    virtual void remove(int fd) = 0;
    // Toggle interests. Implementations skip the syscall when nothing changes.
    virtual void set_read(int fd, bool on) = 0;
    virtual void set_write(int fd, bool on) = 0;
    // Wait up to timeout_ms (-1 forever). Fills `out`, returns count or -1.
    virtual int wait(std::vector<PollEvent>& out, int timeout_ms) = 0;
    virtual const char* name() const = 0;
};

std::unique_ptr<Poller> make_poller();

}  // namespace kv
