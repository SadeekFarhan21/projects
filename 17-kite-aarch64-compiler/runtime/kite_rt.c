/* Kite runtime: the small amount of C that compiled Kite programs link
 * against. Everything is a 64-bit word. Heap objects are never freed
 * (garbage collection is a later milestone).
 *
 * Object layouts (all fields 8 bytes):
 *   string : [len][bytes... NUL]      (pointer points at len)
 *   array  : [len][e0][e1]...
 *   struct : [f0][f1]...              (declaration order)
 *   closure: [code][cap0][cap1]...
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t i64;
typedef struct { i64 len; char bytes[]; } kstr;

extern void _kf_main(void) __asm__("_kf_main");

static void die(const char *msg) {
  fflush(stdout);
  fprintf(stderr, "runtime error: %s\n", msg);
  exit(101);
}

/* Bump allocator over large malloc'd chunks. */
static char *heap_cur = 0, *heap_end = 0;

void *kite_rt_alloc(i64 bytes) {
  bytes = (bytes + 15) & ~(i64)15;
  if (heap_cur == 0 || heap_cur + bytes > heap_end) {
    i64 chunk = bytes > (1 << 20) ? bytes : (1 << 20);
    heap_cur = calloc(1, (size_t)chunk);
    if (!heap_cur) die("out of memory");
    heap_end = heap_cur + chunk;
  }
  void *p = heap_cur;
  heap_cur += bytes;
  return p;
}

static kstr *new_str(i64 len) {
  kstr *s = kite_rt_alloc(8 + len + 1);
  s->len = len;
  s->bytes[len] = 0;
  return s;
}

void kite_rt_print_int(i64 n) { printf("%" PRId64, n); }
void kite_rt_print_bool(i64 b) { fputs(b ? "true" : "false", stdout); }
void kite_rt_print_str(kstr *s) { fwrite(s->bytes, 1, (size_t)s->len, stdout); }
void kite_rt_print_nl(void) { putchar('\n'); }

void kite_rt_oob(i64 idx, i64 len) {
  char buf[128];
  snprintf(buf, sizeof buf, "index out of bounds: index %" PRId64 ", length %" PRId64, idx, len);
  die(buf);
}

void kite_rt_divzero(void) { die("division by zero"); }
void kite_rt_assert_fail(void) { die("assertion failed"); }

i64 *kite_rt_array_new(i64 n, i64 init) {
  if (n < 0) {
    char buf[96];
    snprintf(buf, sizeof buf, "negative array length: %" PRId64, n);
    die(buf);
  }
  i64 *a = kite_rt_alloc(8 * (n + 1));
  a[0] = n;
  for (i64 i = 1; i <= n; i++) a[i] = init;
  return a;
}

kstr *kite_rt_str_concat(kstr *a, kstr *b) {
  kstr *s = new_str(a->len + b->len);
  memcpy(s->bytes, a->bytes, (size_t)a->len);
  memcpy(s->bytes + a->len, b->bytes, (size_t)b->len);
  return s;
}

i64 kite_rt_str_eq(kstr *a, kstr *b) {
  return a->len == b->len && memcmp(a->bytes, b->bytes, (size_t)a->len) == 0;
}

kstr *kite_rt_int_to_str(i64 n) {
  char buf[32];
  int k = snprintf(buf, sizeof buf, "%" PRId64, n);
  kstr *s = new_str(k);
  memcpy(s->bytes, buf, (size_t)k);
  return s;
}

kstr *kite_rt_bool_to_str(i64 b) {
  const char *t = b ? "true" : "false";
  i64 k = (i64)strlen(t);
  kstr *s = new_str(k);
  memcpy(s->bytes, t, (size_t)k);
  return s;
}

kstr *kite_rt_substr(kstr *str, i64 start, i64 count) {
  if (start < 0 || count < 0 || start > str->len || count > str->len - start) {
    char buf[160];
    snprintf(buf, sizeof buf,
             "substr out of bounds: start %" PRId64 ", count %" PRId64 ", length %" PRId64,
             start, count, str->len);
    die(buf);
  }
  kstr *s = new_str(count);
  memcpy(s->bytes, str->bytes + start, (size_t)count);
  return s;
}

kstr *kite_rt_chr(i64 c) {
  if (c < 0 || c > 255) {
    char buf[96];
    snprintf(buf, sizeof buf, "chr argument out of range: %" PRId64, c);
    die(buf);
  }
  kstr *s = new_str(1);
  s->bytes[0] = (char)c;
  return s;
}

int main(void) {
  static char outbuf[1 << 16];
  setvbuf(stdout, outbuf, _IOFBF, sizeof outbuf);
  _kf_main();
  fflush(stdout);
  return 0;
}
