#include <stdint.h>
#include <stdio.h>

int main(void) {
  int64_t n = 400;
  int64_t acc = 0;
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < n; j++)
      for (int64_t k = 0; k < n; k++) {
        acc = acc ^ (i * j + k);
        /* wrapping add, as in Kite (signed overflow is UB in C) */
        acc = (int64_t)((uint64_t)acc + (uint64_t)(acc >> 7));
      }
  printf("%lld\n", (long long)acc);
  return 0;
}
