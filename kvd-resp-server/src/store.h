// The keyspace: string keys to string values, with optional absolute expiry.
//
// Two structures are kept in sync:
//   map_       unordered_map<key, Entry>, the source of truth.
//   ttl_nodes_ a dense vector of pointers to the map nodes that have a TTL.
//
// The dense vector exists so active expiry can sample a uniformly random key
// with a TTL in O(1). unordered_map guarantees that pointers and references to
// its elements stay valid across rehashing (only erase invalidates them), so a
// raw pointer to the node is safe as long as we remove it before erasing.
// Each Entry stores its own index into ttl_nodes_ so removal is an O(1)
// swap-with-last.
//
// All time arguments are absolute Unix time in milliseconds. The store never
// reads a clock itself, which makes expiry deterministic under test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kv {

inline constexpr int64_t kNoExpiry = -1;

struct Entry {
    std::string value;
    int64_t expire_at_ms = kNoExpiry;  // absolute unix ms, or kNoExpiry
    size_t ttl_slot = 0;               // index into Store::ttl_nodes_ when expire_at_ms != kNoExpiry
};

// Transparent hash so lookups by string_view do not allocate a std::string.
struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};

class Store {
public:
    using Map = std::unordered_map<std::string, Entry, StringHash, std::equal_to<>>;
    using Node = Map::value_type;

    Store();

    // Lookup with lazy expiry: an expired key is deleted on access and treated
    // as missing. The returned pointer is valid until the next mutation.
    Entry* find(std::string_view key, int64_t now_ms);

    // Insert or overwrite. expire_at_ms == kNoExpiry clears any TTL unless
    // keep_ttl is true (SET ... KEEPTTL).
    void set(std::string_view key, std::string value, int64_t expire_at_ms, bool keep_ttl = false);

    bool del(std::string_view key, int64_t now_ms);  // true if a live key was removed

    // Set or clear the TTL of an existing live key. Returns false if missing.
    bool set_expiry(std::string_view key, int64_t expire_at_ms, int64_t now_ms);

    // Remaining TTL in ms, -1 if the key has no TTL, -2 if it does not exist.
    int64_t pttl(std::string_view key, int64_t now_ms);

    // Append all live keys matching the glob pattern. O(N), like real KEYS.
    void keys(std::string_view pattern, int64_t now_ms, std::vector<std::string>& out);

    // One active expiry cycle, modelled on Redis's activeExpireCycle: sample up
    // to kSampleSize keys that have a TTL, delete the expired ones, and repeat
    // while more than 25% of the sample was expired and the time budget is not
    // used up. Returns the number of keys deleted.
    struct ExpireStats {
        size_t sampled = 0;
        size_t expired = 0;
        size_t rounds = 0;
    };
    static constexpr size_t kSampleSize = 20;
    ExpireStats active_expire_cycle(int64_t now_ms, int64_t budget_us);

    // Delete every expired key with a full O(N) scan. Used once after AOF
    // replay, where the log can resurrect keys whose deadline passed while
    // the server was down (lazy and active deletions are not logged).
    size_t purge_expired(int64_t now_ms);

    void clear();
    size_t size() const { return map_.size(); }
    size_t ttl_count() const { return ttl_nodes_.size(); }

    // Verify the map/ttl-vector invariants. Used by tests; O(N).
    bool check_invariants(std::string* why = nullptr) const;

private:
    void add_ttl(Node& node, int64_t expire_at_ms);
    void remove_ttl(Node& node);
    void erase(Map::iterator it);

    Map map_;
    std::vector<Node*> ttl_nodes_;
    std::mt19937_64 rng_;
};

}  // namespace kv
