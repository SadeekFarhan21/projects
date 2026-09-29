#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int64_t n = 400;
  int64_t *a = calloc((size_t)(n * n), 8), *b = calloc((size_t)(n * n), 8), *c = calloc((size_t)(n * n), 8);
  for (int64_t i = 0; i < n * n; i++) {
    a[i] = i % 7 - 3;
    b[i] = i % 5 - 2;
  }
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < n; j++) {
      int64_t s = 0;
      for (int64_t k = 0; k < n; k++) s += a[i * n + k] * b[k * n + j];
      c[i * n + j] = s;
    }
  /* wrapping arithmetic, as in Kite: use unsigned to avoid C UB */
  uint64_t check = 0;
  for (int64_t i = 0; i < n * n; i++) check = check * 31 + (uint64_t)c[i];
  printf("%lld\n", (long long)(int64_t)check);
  return 0;
}
