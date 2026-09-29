// Fixed-capacity single-threaded object pool with an intrusive free list of
// indices. acquire/release are O(1) and never touch the system allocator.
// Used by the allocation microbenchmark; in the pipeline itself the ring slots
// play the role of the pool (messages are copied by value into preallocated
// slots), which is why no separate pool appears there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llt {

template <class T>
class ObjectPool {
public:
    explicit ObjectPool(std::size_t n) : slots_(n), next_(n) {
        for (std::size_t i = 0; i < n; ++i) next_[i] = static_cast<uint32_t>(i + 1);
        head_ = 0;
    }
    T* acquire() noexcept {
        if (head_ >= slots_.size()) return nullptr;
        const uint32_t i = head_;
        head_ = next_[i];
        return &slots_[i];
    }
    void release(T* p) noexcept {
        const auto i = static_cast<uint32_t>(p - slots_.data());
        next_[i] = head_;
        head_ = i;
    }

private:
    std::vector<T> slots_;
    std::vector<uint32_t> next_;
    uint32_t head_{0};
};

}  // namespace llt
