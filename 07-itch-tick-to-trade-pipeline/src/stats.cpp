#include "llt/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace llt {

double percentile_sorted(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    // Nearest rank: smallest value with at least q of the mass at or below it.
    std::size_t rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    if (rank == 0) rank = 1;
    if (rank > sorted.size()) rank = sorted.size();
    return sorted[rank - 1];
}

LatencySummary summarize(std::vector<double> v) {
    LatencySummary s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    s.min = v.front();
    s.max = v.back();
    s.p50 = percentile_sorted(v, 0.50);
    s.p90 = percentile_sorted(v, 0.90);
    s.p99 = percentile_sorted(v, 0.99);
    s.p999 = percentile_sorted(v, 0.999);
    return s;
}

std::vector<std::size_t> histogram(const std::vector<double>& samples, double lo, double hi,
                                   double width) {
    const std::size_t nb = static_cast<std::size_t>(std::ceil((hi - lo) / width)) + 1;
    std::vector<std::size_t> h(nb, 0);
    for (double x : samples) {
        std::size_t b;
        if (x < lo) b = 0;
        else if (x >= hi) b = nb - 1;
        else b = static_cast<std::size_t>((x - lo) / width);
        if (b >= nb) b = nb - 1;
        ++h[b];
    }
    return h;
}

std::string format_summary(const LatencySummary& s) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "n=%zu mean=%.0f min=%.0f p50=%.0f p90=%.0f p99=%.0f p99.9=%.0f max=%.0f (ns)",
                  s.n, s.mean, s.min, s.p50, s.p90, s.p99, s.p999, s.max);
    return buf;
}

}  // namespace llt
