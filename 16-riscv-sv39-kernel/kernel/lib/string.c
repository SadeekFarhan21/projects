/* Minimal string and memory routines. The compiler may emit calls to
 * memset/memcpy on its own, so these must exist even if nobody calls them. */
#include "kernel.h"

void *memset(void *dst, int c, size_t n) {
    u8 *d = dst;
    u64 pattern = (u8)c;
    pattern |= pattern << 8;
    pattern |= pattern << 16;
    pattern |= pattern << 32;
    while (n && ((uintptr_t)d & 7)) { *d++ = (u8)c; n--; }
    while (n >= 8) { *(u64 *)d = pattern; d += 8; n -= 8; }
    while (n) { *d++ = (u8)c; n--; }
    return dst;
}

void *memcpy(void *dst, const void *src, size_t n) {
    u8 *d = dst;
    const u8 *s = src;
    if ((((uintptr_t)d | (uintptr_t)s) & 7) == 0) {
        while (n >= 8) { *(u64 *)d = *(const u64 *)s; d += 8; s += 8; n -= 8; }
    }
    while (n) { *d++ = *s++; n--; }
    return dst;
}

void *memmove(void *dst, const void *src, size_t n) {
    u8 *d = dst;
    const u8 *s = src;
    if (d == s || n == 0) return dst;
    if (d < s || d >= s + n) return memcpy(dst, src, n);
    d += n;
    s += n;
    while (n--) *--d = *--s;
    return dst;
}

int memcmp(const void *a, const void *b, size_t n) {
    const u8 *x = a, *y = b;
    for (size_t i = 0; i < n; i++)
        if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}

size_t strlen(const char *s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (u8)*a - (u8)*b;
}

int strncmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i] || !a[i]) return (u8)a[i] - (u8)b[i];
    }
    return 0;
}

size_t strlcpy(char *dst, const char *src, size_t n) {
    size_t len = strlen(src);
    if (n) {
        size_t c = len < n - 1 ? len : n - 1;
        memcpy(dst, src, c);
        dst[c] = 0;
    }
    return len;
}

const char *strstr(const char *hay, const char *needle) {
    size_t n = strlen(needle);
    if (!n) return hay;
    for (; *hay; hay++)
        if (!strncmp(hay, needle, n)) return hay;
    return NULL;
}

/* Kernel command line helpers: "mode=test rxtest crash=panic". */
static const char *find_word(const char *args, const char *w, size_t wl) {
    const char *p = args;
    while (*p) {
        while (*p == ' ') p++;
        const char *start = p;
        while (*p && *p != ' ') p++;
        size_t len = (size_t)(p - start);
        if (len >= wl && !strncmp(start, w, wl) && (len == wl || start[wl] == '='))
            return start;
    }
    return NULL;
}

bool bootarg_has(const char *args, const char *word) {
    const char *s = find_word(args, word, strlen(word));
    return s && (s[strlen(word)] == ' ' || s[strlen(word)] == 0);
}

bool bootarg_value(const char *args, const char *key, char *out, size_t n) {
    size_t kl = strlen(key);
    const char *s = find_word(args, key, kl);
    if (!s || s[kl] != '=') return false;
    s += kl + 1;
    size_t i = 0;
    while (s[i] && s[i] != ' ' && i + 1 < n) { out[i] = s[i]; i++; }
    out[i] = 0;
    return true;
}
