#include "se/corpus.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "se/json.hpp"

namespace se {
namespace {

std::ifstream open_or_throw(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return in;
}

template <typename F>
void for_each_json_line(const std::string& path, F&& f) {
  auto in = open_or_throw(path);
  std::string line, err;
  JsonFields fields;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line == "\r") continue;
    if (!parse_json_string_fields(line, fields, &err))
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": " + err);
    if (json_get(fields, "_id").empty())
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": missing _id");
    f(fields);
  }
}

}  // namespace

std::vector<Document> load_corpus_jsonl(const std::string& path) {
  std::vector<Document> docs;
  for_each_json_line(path, [&](const JsonFields& f) {
    docs.push_back(Document{std::string(json_get(f, "_id")), std::string(json_get(f, "title")),
                            std::string(json_get(f, "text"))});
  });
  return docs;
}

std::vector<Query> load_queries_jsonl(const std::string& path) {
  std::vector<Query> qs;
  for_each_json_line(path, [&](const JsonFields& f) {
    qs.push_back(Query{std::string(json_get(f, "_id")), std::string(json_get(f, "text"))});
  });
  return qs;
}

Qrels load_qrels_tsv(const std::string& path) {
  auto in = open_or_throw(path);
  Qrels q;
  std::string line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string qid, did, score;
    if (!std::getline(ss, qid, '\t') || !std::getline(ss, did, '\t') || !std::getline(ss, score, '\t'))
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": expected 3 tab-separated fields");
    if (lineno == 1 && qid == "query-id") continue;  // BEIR header row
    int rel = 0;
    try {
      rel = std::stoi(score);
    } catch (...) {
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": bad score '" + score + "'");
    }
    if (rel > 0) q[qid][did] = rel;
  }
  return q;
}

}  // namespace se
