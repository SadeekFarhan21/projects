#include "store.h"

#include <algorithm>
#include <chrono>

#include "glob.h"

namespace kv {

Store::Store() : rng_(0x5eed5eedULL) {}

void Store::add_ttl(Node& node, int64_t expire_at_ms) {
    Entry& e = node.second;
    if (e.expire_at_ms == kNoExpiry) {
        e.ttl_slot = ttl_nodes_.size();
        ttl_nodes_.push_back(&node);
    }
    e.expire_at_ms = expire_at_ms;
}

void Store::remove_ttl(Node& node) {
    Entry& e = node.second;
    if (e.expire_at_ms == kNoExpiry) return;
    size_t slot = e.ttl_slot;
    Node* last = ttl_nodes_.back();
    ttl_nodes_[slot] = last;
    last->second.ttl_slot = slot;
    ttl_nodes_.pop_back();
    e.expire_at_ms = kNoExpiry;
    e.ttl_slot = 0;
}

void Store::erase(Map::iterator it) {
    remove_ttl(*it);  // must happen before erase: ttl_nodes_ holds a pointer into the node
    map_.erase(it);
}

Entry* Store::find(std::string_view key, int64_t now_ms) {
    auto it = map_.find(key);
    if (it == map_.end()) return nullptr;
    int64_t exp = it->second.expire_at_ms;
    if (exp != kNoExpiry && exp <= now_ms) {  // lazy expiry
        erase(it);
        return nullptr;
    }
    return &it->second;
}

void Store::set(std::string_view key, std::string value, int64_t expire_at_ms, bool keep_ttl) {
    auto it = map_.find(key);
    if (it == map_.end()) it = map_.emplace(std::string(key), Entry{}).first;
    Node& node = *it;
    node.second.value = std::move(value);
    if (expire_at_ms != kNoExpiry) {
        add_ttl(node, expire_at_ms);
    } else if (!keep_ttl) {
        remove_ttl(node);
    }
}

bool Store::del(std::string_view key, int64_t now_ms) {
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    int64_t exp = it->second.expire_at_ms;
    bool live = !(exp != kNoExpiry && exp <= now_ms);
    erase(it);
    return live;
}

bool Store::set_expiry(std::string_view key, int64_t expire_at_ms, int64_t now_ms) {
    if (!find(key, now_ms)) return false;
    Node& node = *map_.find(key);
    if (expire_at_ms == kNoExpiry)
        remove_ttl(node);
    else
        add_ttl(node, expire_at_ms);
    return true;
}

int64_t Store::pttl(std::string_view key, int64_t now_ms) {
    Entry* e = find(key, now_ms);
    if (!e) return -2;
    if (e->expire_at_ms == kNoExpiry) return -1;
    return e->expire_at_ms - now_ms;
}

void Store::keys(std::string_view pattern, int64_t now_ms, std::vector<std::string>& out) {
    bool all = pattern == "*";
    for (const auto& [k, e] : map_) {
        if (e.expire_at_ms != kNoExpiry && e.expire_at_ms <= now_ms) continue;  // hide, do not delete mid-iteration
        if (all || glob_match(pattern, k)) out.push_back(k);
    }
}

Store::ExpireStats Store::active_expire_cycle(int64_t now_ms, int64_t budget_us) {
    using clk = std::chrono::steady_clock;
    ExpireStats st;
    auto start = clk::now();
    while (!ttl_nodes_.empty()) {
        ++st.rounds;
        size_t n = std::min(kSampleSize, ttl_nodes_.size());
        size_t expired_this_round = 0;
        for (size_t k = 0; k < n && !ttl_nodes_.empty(); ++k) {
            std::uniform_int_distribution<size_t> pick(0, ttl_nodes_.size() - 1);
            Node* node = ttl_nodes_[pick(rng_)];
            ++st.sampled;
            if (node->second.expire_at_ms <= now_ms) {
                auto it = map_.find(node->first);
                erase(it);
                ++expired_this_round;
            }
        }
        st.expired += expired_this_round;
        // Stop when the sampled expired fraction is at most 25%: the remaining
        // garbage is then bounded to roughly a quarter of the TTL keys.
        if (expired_this_round * 4 <= n) break;
        auto used = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - start).count();
        if (used >= budget_us) break;
    }
    return st;
}

size_t Store::purge_expired(int64_t now_ms) {
    size_t n = 0;
    for (auto it = map_.begin(); it != map_.end();) {
        auto next = std::next(it);
        int64_t exp = it->second.expire_at_ms;
        if (exp != kNoExpiry && exp <= now_ms) {
            erase(it);
            ++n;
        }
        it = next;
    }
    return n;
}

void Store::clear() {
    map_.clear();
    ttl_nodes_.clear();
}

bool Store::check_invariants(std::string* why) const {
    auto fail = [&](const char* msg) {
        if (why) *why = msg;
        return false;
    };
    size_t with_ttl = 0;
    for (const auto& node : map_) {
        const Entry& e = node.second;
        if (e.expire_at_ms == kNoExpiry) continue;
        ++with_ttl;
        if (e.ttl_slot >= ttl_nodes_.size()) return fail("ttl_slot out of range");
        if (ttl_nodes_[e.ttl_slot] != &node) return fail("ttl_nodes_[slot] does not point back at node");
    }
    if (with_ttl != ttl_nodes_.size()) return fail("ttl_nodes_ size != number of keys with TTL");
    return true;
}

}  // namespace kv
