// Latency summaries and histograms (off the hot path).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace llt {

struct LatencySummary {
    std::size_t n{0};
    double mean{0}, min{0}, p50{0}, p90{0}, p99{0}, p999{0}, max{0};
};

// Nearest-rank percentile on a sorted copy.
LatencySummary summarize(std::vector<double> samples);

double percentile_sorted(const std::vector<double>& sorted, double q);

// Fixed-width histogram: returns counts for [lo + i*width, lo + (i+1)*width),
// with the final bucket collecting everything >= hi.
std::vector<std::size_t> histogram(const std::vector<double>& samples, double lo, double hi,
                                   double width);

std::string format_summary(const LatencySummary& s);

}  // namespace llt
