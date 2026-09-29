// Gold references: double accumulation, safe formulations, no cleverness.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "gkl/cpu_ref.h"

namespace gkl::cpu {

void sgemm_ref(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
               float* C) {
  std::vector<double> row(N);
  for (int i = 0; i < M; ++i) {
    std::fill(row.begin(), row.end(), 0.0);
    for (int k = 0; k < K; ++k) {
      const double a = A[size_t(i) * K + k];
      const float* b = B + size_t(k) * N;
      for (int j = 0; j < N; ++j) row[j] += a * b[j];
    }
    float* c = C + size_t(i) * N;
    for (int j = 0; j < N; ++j)
      c[j] = float(beta == 0.f ? alpha * row[j] : alpha * row[j] + double(beta) * c[j]);
  }
}

void softmax_ref(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    double m = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < cols; ++c) m = std::max(m, double(xr[c]));
    double d = 0.0;
    for (int c = 0; c < cols; ++c) d += std::exp(double(xr[c]) - m);
    for (int c = 0; c < cols; ++c) yr[c] = float(std::exp(double(xr[c]) - m) / d);
  }
}

void layernorm_ref(const float* x, const float* gamma, const float* beta, float* y, int rows,
                   int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) mean += xr[c];
    mean /= cols;
    double var = 0.0;  // two-pass: exact enough in double
    for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
    var /= cols;
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) yr[c] = float((xr[c] - mean) * rstd * gamma[c] + beta[c]);
  }
}

void rmsnorm_ref(const float* x, const float* gamma, float* y, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    double ss = 0.0;
    for (int c = 0; c < cols; ++c) ss += double(xr[c]) * xr[c];
    const double rrms = 1.0 / std::sqrt(ss / cols + eps);
    for (int c = 0; c < cols; ++c) yr[c] = float(xr[c] * rrms * gamma[c]);
  }
}

double reduce_sum_ref(const float* x, long long n) {
  double s = 0.0;
  for (long long i = 0; i < n; ++i) s += x[i];
  return s;
}

void attention_ref(const float* Q, const float* K, const float* V, float* O, int B, int H, int N,
                   int D, bool causal) {
  const double scale = 1.0 / std::sqrt(double(D));
  std::vector<double> s(N), o(D);
  for (long long bh = 0; bh < (long long)B * H; ++bh) {
    const float* q = Q + bh * N * D;
    const float* k = K + bh * N * D;
    const float* v = V + bh * N * D;
    float* out = O + bh * N * D;
    for (int i = 0; i < N; ++i) {
      const int jmax = causal ? i + 1 : N;
      double m = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < jmax; ++j) {
        double acc = 0.0;
        for (int d = 0; d < D; ++d) acc += double(q[size_t(i) * D + d]) * k[size_t(j) * D + d];
        s[j] = acc * scale;
        m = std::max(m, s[j]);
      }
      double l = 0.0;
      std::fill(o.begin(), o.end(), 0.0);
      for (int j = 0; j < jmax; ++j) {
        const double p = std::exp(s[j] - m);
        l += p;
        for (int d = 0; d < D; ++d) o[d] += p * v[size_t(j) * D + d];
      }
      for (int d = 0; d < D; ++d) out[size_t(i) * D + d] = float(o[d] / l);
    }
  }
}

}  // namespace gkl::cpu
