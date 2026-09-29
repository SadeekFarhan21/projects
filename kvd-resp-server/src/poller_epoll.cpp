// epoll backend (Linux). Compiled only on Linux. Not exercised on the
// development machine (macOS); see README "Known issues".
#if defined(__linux__)

#include <sys/epoll.h>
#include <unistd.h>

#include <cerrno>
#include <unordered_map>

#include "poller.h"

namespace kv {

namespace {

class EpollPoller final : public Poller {
public:
    EpollPoller() : ep_(epoll_create1(EPOLL_CLOEXEC)) {}
    ~EpollPoller() override {
        if (ep_ >= 0) ::close(ep_);
    }

    bool add(int fd) override {
        struct epoll_event ev {};
        ev.events = EPOLLIN;
        ev.data.fd = fd;
        if (epoll_ctl(ep_, EPOLL_CTL_ADD, fd, &ev) != 0) return false;
        state_[fd] = EPOLLIN;
        return true;
    }

    void remove(int fd) override {
        if (state_.erase(fd)) epoll_ctl(ep_, EPOLL_CTL_DEL, fd, nullptr);
    }

    void set_read(int fd, bool on) override { update(fd, EPOLLIN, on); }
    void set_write(int fd, bool on) override { update(fd, EPOLLOUT, on); }

    int wait(std::vector<PollEvent>& out, int timeout_ms) override {
        int n = epoll_wait(ep_, events_, kMaxEvents, timeout_ms);
        out.clear();
        if (n < 0) return errno == EINTR ? 0 : -1;
        for (int i = 0; i < n; ++i) {
            PollEvent pe;
            pe.fd = events_[i].data.fd;
            uint32_t e = events_[i].events;
            pe.readable = e & (EPOLLIN | EPOLLHUP | EPOLLERR);
            pe.writable = e & EPOLLOUT;
            pe.error = e & EPOLLERR;
            out.push_back(pe);
        }
        return n;
    }

    const char* name() const override { return "epoll"; }
    bool ok() const { return ep_ >= 0; }

private:
    static constexpr int kMaxEvents = 1024;

    void update(int fd, uint32_t bit, bool on) {
        auto it = state_.find(fd);
        if (it == state_.end()) return;
        uint32_t next = on ? (it->second | bit) : (it->second & ~bit);
        if (next == it->second) return;
        struct epoll_event ev {};
        ev.events = next;
        ev.data.fd = fd;
        epoll_ctl(ep_, EPOLL_CTL_MOD, fd, &ev);
        it->second = next;
    }

    int ep_;
    std::unordered_map<int, uint32_t> state_;
    struct epoll_event events_[kMaxEvents];
};

}  // namespace

std::unique_ptr<Poller> make_poller() {
    auto p = std::make_unique<EpollPoller>();
    if (!p->ok()) return nullptr;
    return p;
}

}  // namespace kv

#endif
