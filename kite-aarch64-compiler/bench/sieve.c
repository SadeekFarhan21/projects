#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* 8-byte elements to match Kite's word-sized array slots. */
static int64_t sieve(int64_t n) {
  int64_t *composite = calloc((size_t)(n + 1), sizeof(int64_t));
  int64_t count = 0;
  for (int64_t i = 2; i < n + 1; i++) {
    if (!composite[i]) {
      count++;
      for (int64_t j = i * i; j <= n; j += i) composite[j] = 1;
    }
  }
  free(composite);
  return count;
}

int main(void) {
  int64_t total = 0;
  for (int r = 0; r < 10; r++) total += sieve(2000000);
  printf("%lld\n", (long long)total);
  return 0;
}
