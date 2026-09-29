#pragma once
// Minimal JSON reader, just enough for BEIR JSONL records.
// It parses one full JSON value (so malformed input is rejected), and returns
// the top-level string-valued fields of an object. Nested values are skipped.
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace se {

using JsonFields = std::vector<std::pair<std::string, std::string>>;

// Returns false and fills *err on malformed input or a non-object value.
bool parse_json_string_fields(std::string_view json, JsonFields& out, std::string* err = nullptr);

// Convenience lookup; returns empty view if the key is missing.
std::string_view json_get(const JsonFields& f, std::string_view key);

}  // namespace se
