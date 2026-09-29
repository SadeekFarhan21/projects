/* printf for the kernel: %d %i %u %x %X %p %s %c %%, with the l/ll/z length
 * modifiers, a minimum field width, '0' padding and '-' left alignment. */
#include "kernel.h"

struct sink {
    void (*put)(struct sink *, char);
    char *buf;
    size_t cap, len;
};

static void put_console(struct sink *s, char c) { console_putc(c); s->len++; }

static void put_buf(struct sink *s, char c) {
    if (s->len + 1 < s->cap) s->buf[s->len] = c;
    s->len++;
}

static void emit_padded(struct sink *s, const char *str, size_t len, int width,
                        bool left, char padc) {
    int pad = width > (int)len ? width - (int)len : 0;
    /* zero padding goes after a leading '-' sign */
    if (!left && padc == '0' && len && str[0] == '-') {
        s->put(s, '-');
        str++;
        len--;
    }
    if (!left) while (pad-- > 0) s->put(s, padc);
    for (size_t i = 0; i < len; i++) s->put(s, str[i]);
    if (left) while (pad-- > 0) s->put(s, ' ');
}

static size_t fmt_unsigned(char *out, u64 v, int base, bool upper) {
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    char tmp[24];
    size_t n = 0;
    do {
        tmp[n++] = digits[v % (u64)base];
        v /= (u64)base;
    } while (v);
    for (size_t i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    return n;
}

static void vformat(struct sink *s, const char *fmt, va_list ap) {
    char num[32];
    for (; *fmt; fmt++) {
        if (*fmt != '%') { s->put(s, *fmt); continue; }
        fmt++;
        bool left = false;
        char padc = ' ';
        for (;; fmt++) {
            if (*fmt == '-') left = true;
            else if (*fmt == '0') padc = '0';
            else break;
        }
        int width = 0;
        while (*fmt >= '0' && *fmt <= '9') width = width * 10 + (*fmt++ - '0');
        int lng = 0;
        while (*fmt == 'l') { lng++; fmt++; }
        if (*fmt == 'z') { lng = 1; fmt++; }
        if (left) padc = ' ';

        switch (*fmt) {
        case 'd':
        case 'i': {
            i64 v = lng ? va_arg(ap, i64) : va_arg(ap, int);
            size_t n = 0;
            u64 mag = v < 0 ? (u64)(-(v + 1)) + 1 : (u64)v;
            if (v < 0) num[n++] = '-';
            n += fmt_unsigned(num + n, mag, 10, false);
            emit_padded(s, num, n, width, left, padc);
            break;
        }
        case 'u':
        case 'x':
        case 'X': {
            u64 v = lng ? va_arg(ap, u64) : va_arg(ap, unsigned int);
            int base = *fmt == 'u' ? 10 : 16;
            size_t n = fmt_unsigned(num, v, base, *fmt == 'X');
            emit_padded(s, num, n, width, left, padc);
            break;
        }
        case 'p': {
            u64 v = (u64)va_arg(ap, void *);
            num[0] = '0';
            num[1] = 'x';
            char hex[20];
            size_t n = fmt_unsigned(hex, v, 16, false);
            size_t k = 2;
            for (size_t i = n; i < 16; i++) num[k++] = '0';
            for (size_t i = 0; i < n; i++) num[k++] = hex[i];
            emit_padded(s, num, k, width, left, ' ');
            break;
        }
        case 's': {
            const char *str = va_arg(ap, const char *);
            if (!str) str = "(null)";
            emit_padded(s, str, strlen(str), width, left, ' ');
            break;
        }
        case 'c': {
            char c = (char)va_arg(ap, int);
            emit_padded(s, &c, 1, width, left, ' ');
            break;
        }
        case '%':
            s->put(s, '%');
            break;
        case 0:
            return;
        default:
            s->put(s, '%');
            s->put(s, *fmt);
            break;
        }
    }
}

int kvprintf(const char *fmt, va_list ap) {
    struct sink s = {.put = put_console};
    console_lock_acquire();
    vformat(&s, fmt, ap);
    console_lock_release();
    return (int)s.len;
}

int kprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = kvprintf(fmt, ap);
    va_end(ap);
    return n;
}

int ksnprintf(char *buf, size_t n, const char *fmt, ...) {
    struct sink s = {.put = put_buf, .buf = buf, .cap = n};
    va_list ap;
    va_start(ap, fmt);
    vformat(&s, fmt, ap);
    va_end(ap);
    if (n) buf[s.len < n ? s.len : n - 1] = 0;
    return (int)s.len;
}
