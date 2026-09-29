// kqueue backend (macOS, FreeBSD). Read and write are separate filters in
// kqueue, so one fd can produce two events per wait(); the server handles
// each independently.
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)

#include <sys/event.h>
#include <unistd.h>

#include <cerrno>
#include <unordered_map>

#include "poller.h"

namespace kv {

namespace {

class KqueuePoller final : public Poller {
public:
    KqueuePoller() : kq_(kqueue()) {}
    ~KqueuePoller() override {
        if (kq_ >= 0) ::close(kq_);
    }

    bool add(int fd) override {
        if (!change(fd, EVFILT_READ, EV_ADD | EV_ENABLE)) return false;
        state_[fd] = kRead;
        return true;
    }

    void remove(int fd) override {
        auto it = state_.find(fd);
        if (it == state_.end()) return;
        // Closing the fd removes it from kqueue anyway; do it explicitly so
        // remove() before close() is also correct.
        change(fd, EVFILT_READ, EV_DELETE);
        if (it->second & kWrite) change(fd, EVFILT_WRITE, EV_DELETE);
        state_.erase(it);
    }

    void set_read(int fd, bool on) override {
        auto it = state_.find(fd);
        if (it == state_.end() || bool(it->second & kRead) == on) return;
        change(fd, EVFILT_READ, on ? EV_ENABLE : EV_DISABLE);
        it->second = on ? (it->second | kRead) : (it->second & ~kRead);
    }

    void set_write(int fd, bool on) override {
        auto it = state_.find(fd);
        if (it == state_.end() || bool(it->second & kWrite) == on) return;
        change(fd, EVFILT_WRITE, on ? (EV_ADD | EV_ENABLE) : EV_DELETE);
        it->second = on ? (it->second | kWrite) : (it->second & ~kWrite);
    }

    int wait(std::vector<PollEvent>& out, int timeout_ms) override {
        struct timespec ts, *tsp = nullptr;
        if (timeout_ms >= 0) {
            ts.tv_sec = timeout_ms / 1000;
            ts.tv_nsec = static_cast<long>(timeout_ms % 1000) * 1000000L;
            tsp = &ts;
        }
        int n = kevent(kq_, nullptr, 0, events_, kMaxEvents, tsp);
        out.clear();
        if (n < 0) return errno == EINTR ? 0 : -1;
        for (int i = 0; i < n; ++i) {
            const struct kevent& ev = events_[i];
            PollEvent pe;
            pe.fd = static_cast<int>(ev.ident);
            if (ev.flags & EV_ERROR) pe.error = true;
            if (ev.filter == EVFILT_READ) pe.readable = true;  // includes EV_EOF: read() will return 0
            if (ev.filter == EVFILT_WRITE) pe.writable = true;
            out.push_back(pe);
        }
        return n;
    }

    const char* name() const override { return "kqueue"; }

    bool ok() const { return kq_ >= 0; }

private:
    static constexpr unsigned kRead = 1, kWrite = 2;
    static constexpr int kMaxEvents = 1024;

    bool change(int fd, int16_t filter, uint16_t flags) {
        struct kevent ev;
        EV_SET(&ev, fd, filter, flags, 0, 0, nullptr);
        return kevent(kq_, &ev, 1, nullptr, 0, nullptr) == 0;
    }

    int kq_;
    std::unordered_map<int, unsigned> state_;
    struct kevent events_[kMaxEvents];
};

}  // namespace

std::unique_ptr<Poller> make_poller() {
    auto p = std::make_unique<KqueuePoller>();
    if (!p->ok()) return nullptr;
    return p;
}

}  // namespace kv

#endif
