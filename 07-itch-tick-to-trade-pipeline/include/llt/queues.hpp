// Inter-thread queues. All three share the same interface so the pipeline can be
// instantiated with any of them for the ablation:
//
//   SpscRing   lock-free, head and tail on separate cache lines, each side keeps a
//              cached copy of the other side's index (the production choice)
//   SpscNaive  lock-free but head and tail share a cache line, seq_cst atomics,
//              no cached indices (what a first attempt usually looks like)
//   MutexQueue the same fixed ring guarded by std::mutex (isolates lock cost
//              from allocation cost, since none of the three allocate)
//
// Capacity N must be a power of two. Indices are monotonically increasing
// 64-bit counters, so full/empty is (head - tail == N) / (head == tail) with no
// wasted slot and no wraparound ambiguity for 2^64 operations.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>

#include "llt/common.hpp"

namespace llt {

template <class T, std::size_t N>
class SpscRing {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "ring slots are copied with memcpy semantics");

public:
    static constexpr std::size_t capacity = N;

    // Producer side only.
    bool try_push(const T& v) noexcept {
        const uint64_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_cache_ == N) {
            // Looks full through our stale view; refresh once.
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (h - tail_cache_ == N) return false;
        }
        buf_[h & (N - 1)] = v;
        head_.store(h + 1, std::memory_order_release);  // publishes the slot write
        return true;
    }

    // Consumer side only.
    bool try_pop(T& out) noexcept {
        const uint64_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_cache_) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (t == head_cache_) return false;
        }
        out = buf_[t & (N - 1)];
        tail_.store(t + 1, std::memory_order_release);  // hands the slot back
        return true;
    }

    std::size_t size_approx() const noexcept {
        return static_cast<std::size_t>(head_.load(std::memory_order_acquire) -
                                        tail_.load(std::memory_order_acquire));
    }

private:
    // Producer-owned line: the index it writes plus its private cache of tail.
    alignas(kCacheLine) std::atomic<uint64_t> head_{0};
    uint64_t tail_cache_{0};
    // Consumer-owned line.
    alignas(kCacheLine) std::atomic<uint64_t> tail_{0};
    uint64_t head_cache_{0};
    alignas(kCacheLine) T buf_[N];
};

template <class T, std::size_t N>
class SpscNaive {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>);

public:
    static constexpr std::size_t capacity = N;

    bool try_push(const T& v) noexcept {
        const uint64_t h = head_.load();
        if (h - tail_.load() == N) return false;
        buf_[h & (N - 1)] = v;
        head_.store(h + 1);
        return true;
    }

    bool try_pop(T& out) noexcept {
        const uint64_t t = tail_.load();
        if (t == head_.load()) return false;
        out = buf_[t & (N - 1)];
        tail_.store(t + 1);
        return true;
    }

private:
    // Deliberately adjacent: every push invalidates the consumer's copy of the
    // line holding tail and vice versa.
    std::atomic<uint64_t> head_{0};
    std::atomic<uint64_t> tail_{0};
    T buf_[N];
};

template <class T, std::size_t N>
class MutexQueue {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "capacity must be a power of two");

public:
    static constexpr std::size_t capacity = N;

    bool try_push(const T& v) noexcept {
        std::lock_guard<std::mutex> g(m_);
        if (head_ - tail_ == N) return false;
        buf_[head_ & (N - 1)] = v;
        ++head_;
        return true;
    }

    bool try_pop(T& out) noexcept {
        std::lock_guard<std::mutex> g(m_);
        if (head_ == tail_) return false;
        out = buf_[tail_ & (N - 1)];
        ++tail_;
        return true;
    }

private:
    std::mutex m_;
    uint64_t head_{0};
    uint64_t tail_{0};
    T buf_[N];
};

// Blocking helpers used by the pipeline. They spin; nothing on the hot path
// ever sleeps. Returns the number of failed attempts (backpressure signal).
template <class Q, class T>
inline uint64_t push_spin(Q& q, const T& v) noexcept {
    uint64_t spins = 0;
    while (!q.try_push(v)) {
        ++spins;
        cpu_relax();
    }
    return spins;
}

template <class Q, class T>
inline void pop_spin(Q& q, T& out) noexcept {
    while (!q.try_pop(out)) cpu_relax();
}

}  // namespace llt
