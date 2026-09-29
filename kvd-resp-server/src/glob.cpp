#include "glob.h"

#include <cctype>
#include <utility>

namespace kv {

namespace {

char fold(char c, bool nocase) {
    return nocase ? static_cast<char>(std::tolower(static_cast<unsigned char>(c))) : c;
}

// Match a [...] class starting at pattern[p] == '['. Sets `p` to the index of
// the closing ']' (or the end of the pattern if unterminated). Returns whether
// `c` is in the class.
bool match_class(std::string_view pat, size_t& p, char c, bool nocase) {
    ++p;  // skip '['
    bool negate = false;
    if (p < pat.size() && pat[p] == '^') {
        negate = true;
        ++p;
    }
    bool matched = false;
    while (p < pat.size() && pat[p] != ']') {
        if (pat[p] == '\\' && p + 1 < pat.size()) {
            ++p;
            if (fold(pat[p], nocase) == fold(c, nocase)) matched = true;
        } else if (p + 2 < pat.size() && pat[p + 1] == '-' && pat[p + 2] != ']') {
            char lo = fold(pat[p], nocase), hi = fold(pat[p + 2], nocase);
            if (lo > hi) std::swap(lo, hi);
            char cc = fold(c, nocase);
            if (cc >= lo && cc <= hi) matched = true;
            p += 2;
        } else if (fold(pat[p], nocase) == fold(c, nocase)) {
            matched = true;
        }
        ++p;
    }
    return negate ? !matched : matched;
}

}  // namespace

// Iterative matcher with single-star backtracking: O(|pattern| * |str|) worst
// case instead of the exponential recursion a naive implementation has on
// patterns like "a*a*a*a*b".
bool glob_match(std::string_view pat, std::string_view s, bool nocase) {
    size_t p = 0, i = 0;
    size_t star_p = std::string_view::npos, star_i = 0;
    while (i < s.size()) {
        if (p < pat.size()) {
            char pc = pat[p];
            if (pc == '*') {
                while (p < pat.size() && pat[p] == '*') ++p;  // collapse runs of '*'
                if (p == pat.size()) return true;
                star_p = p;
                star_i = i;
                continue;
            }
            if (pc == '?') {
                ++p;
                ++i;
                continue;
            }
            if (pc == '[') {
                size_t q = p;
                if (match_class(pat, q, s[i], nocase)) {
                    p = (q < pat.size()) ? q + 1 : q;
                    ++i;
                    continue;
                }
            } else {
                char lit = pc;
                size_t adv = 1;
                if (pc == '\\' && p + 1 < pat.size()) {
                    lit = pat[p + 1];
                    adv = 2;
                }
                if (fold(lit, nocase) == fold(s[i], nocase)) {
                    p += adv;
                    ++i;
                    continue;
                }
            }
        }
        // Mismatch: backtrack to the last '*' and let it swallow one more char.
        if (star_p == std::string_view::npos) return false;
        p = star_p;
        i = ++star_i;
    }
    while (p < pat.size() && pat[p] == '*') ++p;
    return p == pat.size();
}

}  // namespace kv
