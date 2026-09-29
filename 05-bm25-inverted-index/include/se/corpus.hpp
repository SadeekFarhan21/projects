#pragma once
// Loaders for the BEIR on-disk format: corpus.jsonl, queries.jsonl, qrels/*.tsv.
#include <string>
#include <unordered_map>
#include <vector>

namespace se {

struct Document {
  std::string id;
  std::string title;
  std::string text;
};

struct Query {
  std::string id;
  std::string text;
};

// qrels[query_id][doc_id] = graded relevance (only > 0 entries are kept).
using Qrels = std::unordered_map<std::string, std::unordered_map<std::string, int>>;

// All loaders throw std::runtime_error with file:line context on bad input.
std::vector<Document> load_corpus_jsonl(const std::string& path);
std::vector<Query> load_queries_jsonl(const std::string& path);
Qrels load_qrels_tsv(const std::string& path);

}  // namespace se
