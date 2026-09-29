// Open-addressing hash map from OrderId (uint64, 0 reserved) to a uint32
// slot index. Linear probing, power-of-two capacity, load factor <= 1/2,
// and backward-shift deletion so there are no tombstones to accumulate.
//
// This replaced std::unordered_map on the cancel path; see DEVLOG.md.
#pragma once

#include "exchange/types.hpp"

#include <cstdint>
#include <vector>

namespace exch {

class IdMap {
public:
    static constexpr std::uint32_t kMissing = 0xFFFFFFFFu;

    explicit IdMap(std::size_t expected = 1024) { rehash(capacity_for(expected)); }

    std::size_t size() const { return size_; }

    std::uint32_t find(OrderId key) const {
        std::size_t i = home(key);
        while (true) {
            const Slot& s = slots_[i];
            if (s.key == key) return s.value;
            if (s.key == 0) return kMissing;
            i = (i + 1) & mask_;
        }
    }

    // Returns false if the key already exists (value is left unchanged).
    bool insert(OrderId key, std::uint32_t value) {
        if ((size_ + 1) * 2 > slots_.size()) rehash(slots_.size() * 2);
        std::size_t i = home(key);
        while (slots_[i].key != 0) {
            if (slots_[i].key == key) return false;
            i = (i + 1) & mask_;
        }
        slots_[i] = Slot{key, value};
        ++size_;
        return true;
    }

    bool erase(OrderId key) {
        std::size_t i = home(key);
        while (true) {
            if (slots_[i].key == 0) return false;
            if (slots_[i].key == key) break;
            i = (i + 1) & mask_;
        }
        // Backward shift: pull later entries of the probe run into the hole
        // when their home position allows it.
        std::size_t hole = i;
        std::size_t j = i;
        while (true) {
            j = (j + 1) & mask_;
            if (slots_[j].key == 0) break;
            std::size_t h = home(slots_[j].key);
            // Entry at j may move to `hole` iff hole lies cyclically in [h, j).
            bool movable = (hole <= j) ? (h <= hole || h > j) : (h <= hole && h > j);
            if (movable) {
                slots_[hole] = slots_[j];
                hole = j;
            }
        }
        slots_[hole] = Slot{};
        --size_;
        return true;
    }

    void clear() {
        for (auto& s : slots_) s = Slot{};
        size_ = 0;
    }

private:
    struct Slot {
        OrderId key{0};
        std::uint32_t value{0};
    };

    static std::size_t capacity_for(std::size_t n) {
        std::size_t c = 16;
        while (c < n * 2) c <<= 1;
        return c;
    }

    std::size_t home(OrderId key) const {
#ifdef EXCH_IDMAP_IDENTITY_HASH
        // Ablation only: sequential ids land in sequential slots (great
        // locality), but ids sharing low bits all collide.
        return static_cast<std::size_t>(key) & mask_;
#endif
        // splitmix64 finalizer: robust to any id pattern a client picks.
        std::uint64_t x = key;
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<std::size_t>(x) & mask_;
    }

    void rehash(std::size_t cap) {
        std::vector<Slot> old = std::move(slots_);
        slots_.assign(cap, Slot{});
        mask_ = cap - 1;
        size_ = 0;
        for (const Slot& s : old)
            if (s.key != 0) insert(s.key, s.value);
    }

    std::vector<Slot> slots_;
    std::size_t mask_{0};
    std::size_t size_{0};
};

} // namespace exch
