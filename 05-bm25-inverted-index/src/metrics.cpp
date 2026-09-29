#include "se/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace se {

namespace {
int grade(const Judgments& rel, const std::string& d) {
  auto it = rel.find(d);
  return it == rel.end() ? 0 : it->second;
}
}  // namespace

double ndcg_at_k(const std::vector<std::string>& ranked, const Judgments& rel, size_t k) {
  double dcg = 0.0;
  for (size_t i = 0; i < ranked.size() && i < k; ++i)
    dcg += grade(rel, ranked[i]) / std::log2(static_cast<double>(i) + 2.0);
  std::vector<int> ideal;
  for (const auto& [d, g] : rel)
    if (g > 0) ideal.push_back(g);
  std::sort(ideal.begin(), ideal.end(), std::greater<>());
  double idcg = 0.0;
  for (size_t i = 0; i < ideal.size() && i < k; ++i)
    idcg += ideal[i] / std::log2(static_cast<double>(i) + 2.0);
  return idcg > 0 ? dcg / idcg : 0.0;
}

double mrr_at_k(const std::vector<std::string>& ranked, const Judgments& rel, size_t k) {
  for (size_t i = 0; i < ranked.size() && i < k; ++i)
    if (grade(rel, ranked[i]) > 0) return 1.0 / static_cast<double>(i + 1);
  return 0.0;
}

double recall_at_k(const std::vector<std::string>& ranked, const Judgments& rel, size_t k) {
  size_t total = 0;
  for (const auto& [d, g] : rel)
    if (g > 0) ++total;
  if (total == 0) return 0.0;
  size_t hit = 0;
  for (size_t i = 0; i < ranked.size() && i < k; ++i)
    if (grade(rel, ranked[i]) > 0) ++hit;
  return static_cast<double>(hit) / static_cast<double>(total);
}

}  // namespace se
