#include "se/json.hpp"

#include <cstdint>

namespace se {
namespace {

struct Parser {
  std::string_view s;
  size_t i = 0;
  std::string err;

  bool fail(const char* msg) {
    if (err.empty()) err = std::string(msg) + " at offset " + std::to_string(i);
    return false;
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  static void put_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  bool hex4(uint32_t& v) {
    if (i + 4 > s.size()) return fail("truncated \\u escape");
    v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else return fail("bad hex digit");
    }
    return true;
  }
  // Parses a string starting at s[i] == '"'. If out is null the value is skipped.
  bool str(std::string* out) {
    if (i >= s.size() || s[i] != '"') return fail("expected string");
    ++i;
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') return true;
      if (static_cast<unsigned char>(c) < 0x20) return fail("control character in string");
      if (c != '\\') {
        if (out) out->push_back(c);
        continue;
      }
      if (i >= s.size()) break;
      char e = s[i++];
      char lit = 0;
      switch (e) {
        case '"': lit = '"'; break;
        case '\\': lit = '\\'; break;
        case '/': lit = '/'; break;
        case 'b': lit = '\b'; break;
        case 'f': lit = '\f'; break;
        case 'n': lit = '\n'; break;
        case 'r': lit = '\r'; break;
        case 't': lit = '\t'; break;
        case 'u': {
          uint32_t cp;
          if (!hex4(cp)) return false;
          // Combine UTF-16 surrogate pairs; a lone surrogate becomes U+FFFD.
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            uint32_t lo = 0;
            if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
              size_t save = i;
              i += 2;
              if (!hex4(lo)) return false;
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                i = save;
                cp = 0xFFFD;
              }
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD;
          }
          if (out) put_utf8(cp, *out);
          continue;
        }
        default: return fail("bad escape");
      }
      if (out) out->push_back(lit);
    }
    return fail("unterminated string");
  }
  bool literal(std::string_view w) {
    if (s.substr(i, w.size()) != w) return fail("bad literal");
    i += w.size();
    return true;
  }
  bool number() {
    size_t start = i;
    if (i < s.size() && s[i] == '-') ++i;
    while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
                            s[i] == 'E' || s[i] == '+' || s[i] == '-'))
      ++i;
    if (i == start) return fail("expected value");
    return true;
  }
  bool value(int depth, JsonFields* top) {
    if (depth > 64) return fail("nesting too deep");
    ws();
    if (i >= s.size()) return fail("unexpected end");
    char c = s[i];
    if (c == '{') {
      ++i;
      ws();
      if (i < s.size() && s[i] == '}') { ++i; return true; }
      for (;;) {
        ws();
        std::string key;
        if (!str(&key)) return false;
        ws();
        if (i >= s.size() || s[i] != ':') return fail("expected ':'");
        ++i;
        ws();
        if (top && i < s.size() && s[i] == '"') {
          std::string v;
          if (!str(&v)) return false;
          top->emplace_back(std::move(key), std::move(v));
        } else if (!value(depth + 1, nullptr)) {
          return false;
        }
        ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        return fail("expected ',' or '}'");
      }
    }
    if (top) return fail("top-level value is not an object");
    if (c == '[') {
      ++i;
      ws();
      if (i < s.size() && s[i] == ']') { ++i; return true; }
      for (;;) {
        if (!value(depth + 1, nullptr)) return false;
        ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == ']') { ++i; return true; }
        return fail("expected ',' or ']'");
      }
    }
    if (c == '"') return str(nullptr);
    if (c == 't') return literal("true");
    if (c == 'f') return literal("false");
    if (c == 'n') return literal("null");
    return number();
  }
};

}  // namespace

bool parse_json_string_fields(std::string_view json, JsonFields& out, std::string* err) {
  out.clear();
  Parser p;
  p.s = json;
  bool ok = p.value(0, &out);
  if (ok) {
    p.ws();
    if (p.i != json.size()) ok = p.fail("trailing characters");
  }
  if (!ok && err) *err = p.err;
  return ok;
}

std::string_view json_get(const JsonFields& f, std::string_view key) {
  for (const auto& [k, v] : f)
    if (k == key) return v;
  return {};
}

}  // namespace se
