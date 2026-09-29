#pragma once
// Ranking metrics, matching trec_eval / pytrec_eval conventions:
//  * nDCG uses linear gain (gain = relevance grade) and log2(rank + 1) discount;
//    the ideal ranking is built from all judged relevant documents.
//  * MRR@k is the reciprocal rank of the first relevant hit within k, else 0.
//  * Recall@k is |relevant in top k| / |relevant|.
#include <string>
#include <unordered_map>
#include <vector>

namespace se {

using Judgments = std::unordered_map<std::string, int>;  // doc id -> grade > 0

double ndcg_at_k(const std::vector<std::string>& ranked, const Judgments& rel, size_t k);
double mrr_at_k(const std::vector<std::string>& ranked, const Judgments& rel, size_t k);
double recall_at_k(const std::vector<std::string>& ranked, const Judgments& rel, size_t k);

}  // namespace se
