// Redis-compatible glob matching for KEYS: *, ?, [abc], [^a-z], and \ escapes.
#pragma once

#include <string_view>

namespace kv {

bool glob_match(std::string_view pattern, std::string_view str, bool nocase = false);

}  // namespace kv
