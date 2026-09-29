// Float CPU versions of the GPU algorithms, plus deliberately naive variants.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "gkl/cpu_ref.h"

namespace gkl::cpu {

// Cache-blocked i-k-j GEMM. The inner j loop is contiguous in B and C so the
// compiler vectorizes it (NEON on the M4). Single threaded on purpose: it is a
// CPU baseline for the benchmark table, not a competitor to Accelerate.
void sgemm_cpu_blocked(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                       float* C) {
  constexpr int MB = 64, KB = 256, NB = 512;
  std::vector<float> acc(size_t(M) * N, 0.f);
  for (int i0 = 0; i0 < M; i0 += MB)
    for (int k0 = 0; k0 < K; k0 += KB)
      for (int j0 = 0; j0 < N; j0 += NB) {
        const int i1 = std::min(M, i0 + MB), k1 = std::min(K, k0 + KB), j1 = std::min(N, j0 + NB);
        for (int i = i0; i < i1; ++i) {
          float* crow = acc.data() + size_t(i) * N;
          for (int k = k0; k < k1; ++k) {
            const float a = A[size_t(i) * K + k];
            const float* brow = B + size_t(k) * N;
            for (int j = j0; j < j1; ++j) crow[j] += a * brow[j];
          }
        }
      }
  for (size_t i = 0; i < size_t(M) * N; ++i)
    C[i] = beta == 0.f ? alpha * acc[i] : alpha * acc[i] + beta * C[i];
}

void softmax_online_cpu(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    float m = -INFINITY, d = 0.f;
    for (int c = 0; c < cols; ++c) {
      const float v = xr[c];
      if (v > m) {
        d = d * std::exp(m - v) + 1.f;
        m = v;
      } else if (m != -INFINITY) {
        d += std::exp(v - m);
      }
    }
    const float inv = 1.f / d;
    for (int c = 0; c < cols; ++c) yr[c] = std::exp(xr[c] - m) * inv;
  }
}

void softmax_unsafe_cpu(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    float d = 0.f;
    for (int c = 0; c < cols; ++c) d += std::exp(xr[c]);
    for (int c = 0; c < cols; ++c) yr[c] = std::exp(xr[c]) / d;
  }
}

void layernorm_welford_cpu(const float* x, const float* gamma, const float* beta, float* y,
                           int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    float n = 0.f, mean = 0.f, m2 = 0.f;
    for (int c = 0; c < cols; ++c) {
      n += 1.f;
      const float delta = xr[c] - mean;
      mean += delta / n;
      m2 += delta * (xr[c] - mean);
    }
    const float rstd = 1.f / std::sqrt(m2 / cols + eps);
    for (int c = 0; c < cols; ++c) yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
  }
}

void layernorm_onepass_cpu(const float* x, const float* gamma, const float* beta, float* y,
                           int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * cols;
    float* yr = y + size_t(r) * cols;
    float s = 0.f, ss = 0.f;
    for (int c = 0; c < cols; ++c) {
      s += xr[c];
      ss += xr[c] * xr[c];
    }
    const float mean = s / cols;
    const float var = std::max(0.f, ss / cols - mean * mean);  // clamp: can go negative
    const float rstd = 1.f / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
  }
}

float reduce_sum_sequential_cpu(const float* x, long long n) {
  float s = 0.f;
  for (long long i = 0; i < n; ++i) s += x[i];
  return s;
}

float reduce_sum_tree_cpu(const float* x, long long n, int lanes) {
  std::vector<float> acc(lanes, 0.f);
  for (long long i = 0; i < n; ++i) acc[i % lanes] += x[i];
  for (int width = 1; width < lanes; width *= 2)  // pairwise tree, like the shuffles
    for (int i = 0; i + width < lanes; i += 2 * width) acc[i] += acc[i + width];
  return acc[0];
}

void attention_flash_cpu(const float* Q, const float* K, const float* V, float* O, int B, int H,
                         int N, int D, bool causal, int br, int bc) {
  const float scale = 1.f / std::sqrt(float(D));
  std::vector<float> s(size_t(br) * bc), m(br), l(br), o(size_t(br) * D);
  for (long long bh = 0; bh < (long long)B * H; ++bh) {
    const float* q = Q + bh * N * D;
    const float* k = K + bh * N * D;
    const float* v = V + bh * N * D;
    float* out = O + bh * N * D;
    for (int q0 = 0; q0 < N; q0 += br) {
      const int rows = std::min(br, N - q0);
      std::fill(m.begin(), m.end(), -INFINITY);
      std::fill(l.begin(), l.end(), 0.f);
      std::fill(o.begin(), o.end(), 0.f);
      const int kv_end = causal ? std::min(N, q0 + br) : N;
      for (int k0 = 0; k0 < kv_end; k0 += bc) {
        const int cols = std::min(bc, N - k0);
        for (int r = 0; r < rows; ++r) {
          const int qi = q0 + r;
          float tile_max = -INFINITY;
          for (int j = 0; j < cols; ++j) {
            float acc = 0.f;
            for (int d = 0; d < D; ++d) acc += q[size_t(qi) * D + d] * k[size_t(k0 + j) * D + d];
            const bool valid = !causal || k0 + j <= qi;
            s[size_t(r) * bc + j] = valid ? acc * scale : -INFINITY;
            tile_max = std::max(tile_max, s[size_t(r) * bc + j]);
          }
          const float m_new = std::max(m[r], tile_max);
          if (m_new == -INFINITY) continue;  // fully masked so far
          const float alpha = m[r] == -INFINITY ? 0.f : std::exp(m[r] - m_new);
          float psum = 0.f;
          float* orow = o.data() + size_t(r) * D;
          for (int d = 0; d < D; ++d) orow[d] *= alpha;
          for (int j = 0; j < cols; ++j) {
            const float p = std::exp(s[size_t(r) * bc + j] - m_new);
            psum += p;
            for (int d = 0; d < D; ++d) orow[d] += p * v[size_t(k0 + j) * D + d];
          }
          l[r] = l[r] * alpha + psum;
          m[r] = m_new;
        }
      }
      for (int r = 0; r < rows; ++r)
        for (int d = 0; d < D; ++d) out[size_t(q0 + r) * D + d] = o[size_t(r) * D + d] / l[r];
    }
  }
}

void attention_naive_cpu(const float* Q, const float* K, const float* V, float* O, int B, int H,
                         int N, int D, bool causal) {
  const float scale = 1.f / std::sqrt(float(D));
  std::vector<float> S(size_t(N) * N);  // the quadratic buffer
  for (long long bh = 0; bh < (long long)B * H; ++bh) {
    const float* q = Q + bh * N * D;
    const float* k = K + bh * N * D;
    const float* v = V + bh * N * D;
    float* out = O + bh * N * D;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        float acc = 0.f;
        for (int d = 0; d < D; ++d) acc += q[size_t(i) * D + d] * k[size_t(j) * D + d];
        S[size_t(i) * N + j] = (causal && j > i) ? -INFINITY : acc * scale;
      }
    softmax_online_cpu(S.data(), S.data(), N, N);
    for (int i = 0; i < N; ++i) {
      float* orow = out + size_t(i) * D;
      std::fill(orow, orow + D, 0.f);
      for (int j = 0; j < N; ++j) {
        const float p = S[size_t(i) * N + j];
        for (int d = 0; d < D; ++d) orow[d] += p * v[size_t(j) * D + d];
      }
    }
  }
}

}  // namespace gkl::cpu
