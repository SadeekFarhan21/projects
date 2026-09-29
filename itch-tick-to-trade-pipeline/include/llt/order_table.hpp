// Order-reference to resting-order lookup tables.
//
// FlatOrderMap is an open-addressing hash table with linear probing and
// backward-shift deletion (no tombstones, so probe lengths do not degrade over a
// trading day of churn). All memory is allocated in the constructor.
//
// StdOrderMap wraps std::unordered_map with the same interface. It allocates a
// node on every insert and frees it on erase; it exists for the ablation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace llt {

struct OrderInfo {
    uint32_t price{0};
    uint32_t qty{0};
    uint16_t locate{0};
    char side{0};
    bool in_band{true};  // false if the price was outside the ladder window
};

class FlatOrderMap {
public:
    // capacity is rounded up to a power of two. Keep the load factor below
    // ~0.5 for short probe sequences; insert fails (returns false) when full.
    explicit FlatOrderMap(std::size_t capacity = 1u << 20) {
        std::size_t c = 16;
        while (c < capacity) c <<= 1;
        slots_.assign(c, Slot{});
        mask_ = c - 1;
        shift_ = 64 - static_cast<unsigned>(__builtin_ctzll(c));
    }

    OrderInfo* find(uint64_t key) noexcept {
        for (std::size_t i = home(key);; i = (i + 1) & mask_) {
            Slot& s = slots_[i];
            if (s.key == key) return &s.v;
            if (s.key == kEmpty) return nullptr;
        }
    }

    // key 0 is reserved as the empty marker (ITCH order refs start at 1).
    bool insert(uint64_t key, const OrderInfo& v) noexcept {
        if (key == kEmpty || size_ + 1 > mask_) return false;  // keep >= 1 empty slot
        for (std::size_t i = home(key);; i = (i + 1) & mask_) {
            Slot& s = slots_[i];
            if (s.key == key) return false;
            if (s.key == kEmpty) {
                s.key = key;
                s.v = v;
                ++size_;
                return true;
            }
        }
    }

    bool erase(uint64_t key) noexcept {
        std::size_t i = home(key);
        for (;; i = (i + 1) & mask_) {
            if (slots_[i].key == key) break;
            if (slots_[i].key == kEmpty) return false;
        }
        // Backward shift: pull later entries of the same probe cluster into the
        // hole if their home slot is not cyclically inside (hole, j].
        std::size_t hole = i;
        for (std::size_t j = (hole + 1) & mask_; slots_[j].key != kEmpty; j = (j + 1) & mask_) {
            const std::size_t h = home(slots_[j].key);
            const bool h_in_range = (hole <= j) ? (h > hole && h <= j) : (h > hole || h <= j);
            if (!h_in_range) {
                slots_[hole] = slots_[j];
                hole = j;
            }
        }
        slots_[hole].key = kEmpty;
        --size_;
        return true;
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return mask_ + 1; }

private:
    static constexpr uint64_t kEmpty = 0;
    struct Slot {
        uint64_t key{kEmpty};
        OrderInfo v{};
    };
    // Fibonacci hashing: multiply and keep the top bits. Sequential ITCH order
    // refs spread evenly across the table.
    std::size_t home(uint64_t k) const noexcept {
        return static_cast<std::size_t>((k * 0x9E3779B97F4A7C15ull) >> shift_);
    }
    std::vector<Slot> slots_;
    std::size_t mask_{0};
    std::size_t size_{0};
    unsigned shift_{0};
};

class StdOrderMap {
public:
    explicit StdOrderMap(std::size_t capacity = 1u << 20) { m_.reserve(capacity / 2); }
    OrderInfo* find(uint64_t key) noexcept {
        auto it = m_.find(key);
        return it == m_.end() ? nullptr : &it->second;
    }
    bool insert(uint64_t key, const OrderInfo& v) { return m_.emplace(key, v).second; }
    bool erase(uint64_t key) { return m_.erase(key) == 1; }
    std::size_t size() const noexcept { return m_.size(); }

private:
    std::unordered_map<uint64_t, OrderInfo> m_;
};

}  // namespace llt
