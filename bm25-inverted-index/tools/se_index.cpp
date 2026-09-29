// Build an index from a BEIR corpus.jsonl.
//
//   se_index --corpus data/scifact/corpus.jsonl --out build/scifact.idx
//            [--no-stem] [--no-stop] [--threads N] [--stats-json out.json]
#include <algorithm>
#include <cstdio>
#include <thread>

#include "cli.hpp"
#include "se/corpus.hpp"
#include "se/index.hpp"
#include "se/index_builder.hpp"

int main(int argc, char** argv) {
  auto a = cli::parse(argc, argv, {"no-stem", "no-stop"});
  const std::string corpus = a.get("corpus"), out = a.get("out");
  if (corpus.empty() || out.empty())
    cli::die("usage: se_index --corpus corpus.jsonl --out index.idx [--no-stem] [--no-stop] "
             "[--threads N] [--stats-json f]");
  se::TokenizerOptions opts;
  opts.stem = !a.has("no-stem");
  opts.stopwords = !a.has("no-stop");
  unsigned threads = static_cast<unsigned>(
      a.num("threads", std::max(1u, std::thread::hardware_concurrency())));
  threads = std::max(1u, threads);

  try {
    const double load0 = cli::load_avg_1m();
    const double c0 = cli::process_cpu_s();
    double t0 = cli::now_s();
    auto docs = se::load_corpus_jsonl(corpus);
    double t_load = cli::now_s() - t0;
    double c_load = cli::process_cpu_s() - c0;

    se::IndexBuilder b(opts);
    double t_tok = 0, t_inv = 0, c_tok = 0, c_inv = 0;
    const size_t batch = 8192;
    std::vector<std::vector<se::Token>> toks(batch);
    for (size_t start = 0; start < docs.size(); start += batch) {
      const size_t n = std::min(batch, docs.size() - start);
      double s0 = cli::now_s(), k0 = cli::process_cpu_s();
      // Tokenize the batch in parallel: each worker takes a strided slice.
      auto work = [&](unsigned w) {
        for (size_t i = w; i < n; i += threads)
          toks[i] = se::IndexBuilder::tokenize_document(b.tokenizer(), docs[start + i].title,
                                                        docs[start + i].text);
      };
      if (threads == 1) {
        work(0);
      } else {
        std::vector<std::thread> pool;
        for (unsigned w = 0; w < threads; ++w) pool.emplace_back(work, w);
        for (auto& th : pool) th.join();
      }
      double s1 = cli::now_s(), k1 = cli::process_cpu_s();
      for (size_t i = 0; i < n; ++i) b.add_tokenized(docs[start + i].id, toks[i]);
      t_inv += cli::now_s() - s1;
      c_inv += cli::process_cpu_s() - k1;
      t_tok += s1 - s0;
      c_tok += k1 - k0;
    }
    double t1 = cli::now_s(), k2 = cli::process_cpu_s();
    auto st = b.write(out);
    double t_write = cli::now_s() - t1;
    double c_write = cli::process_cpu_s() - k2;
    double t2 = cli::now_s();
    auto ix = se::Index::open(out);  // round-trip check: validates every list
    double t_open = cli::now_s() - t2;
    double total = t_load + t_tok + t_inv + t_write;

    double ratio = static_cast<double>(st.raw_doc_stream_bytes + st.raw_pos_stream_bytes) /
                   static_cast<double>(std::max<uint64_t>(1, st.doc_stream_bytes + st.pos_stream_bytes));
    std::printf("docs=%u terms=%u postings=%llu positions=%llu\n", st.num_docs, st.num_terms,
                (unsigned long long)st.num_postings, (unsigned long long)st.num_positions);
    std::printf("index=%s bytes=%llu (doc stream %llu, pos stream %llu; raw uint32 would be %llu + %llu, ratio %.2fx)\n",
                out.c_str(), (unsigned long long)st.file_bytes,
                (unsigned long long)st.doc_stream_bytes, (unsigned long long)st.pos_stream_bytes,
                (unsigned long long)st.raw_doc_stream_bytes,
                (unsigned long long)st.raw_pos_stream_bytes, ratio);
    std::printf("wall: load %.3fs tokenize %.3fs (%u threads) invert %.3fs write %.3fs total %.3fs; open+validate %.3fs\n",
                t_load, t_tok, threads, t_inv, t_write, total, t_open);
    std::printf("cpu:  load %.3fs tokenize %.3fs invert %.3fs write %.3fs  (load avg %.1f)\n", c_load,
                c_tok, c_inv, c_write, load0);

    if (a.has("stats-json")) {
      FILE* f = std::fopen(a.get("stats-json").c_str(), "w");
      if (!f) cli::die("cannot write stats json");
      std::fprintf(f,
                   "{\n  \"corpus\": \"%s\",\n  \"stem\": %s,\n  \"stopwords\": %s,\n  \"threads\": %u,\n"
                   "  \"num_docs\": %u,\n  \"num_terms\": %u,\n  \"num_postings\": %llu,\n"
                   "  \"num_positions\": %llu,\n  \"avg_doc_len\": %.3f,\n  \"file_bytes\": %llu,\n"
                   "  \"doc_stream_bytes\": %llu,\n  \"pos_stream_bytes\": %llu,\n"
                   "  \"raw_doc_stream_bytes\": %llu,\n  \"raw_pos_stream_bytes\": %llu,\n"
                   "  \"lexicon_bytes\": %llu,\n  \"doc_table_bytes\": %llu,\n"
                   "  \"postings_compression_ratio\": %.4f,\n"
                   "  \"bits_per_posting_doc_stream\": %.4f,\n  \"bits_per_position\": %.4f,\n"
                   "  \"seconds_load\": %.4f,\n  \"seconds_tokenize\": %.4f,\n  \"seconds_invert\": %.4f,\n"
                   "  \"seconds_write\": %.4f,\n  \"seconds_total\": %.4f,\n  \"seconds_open_validate\": %.4f,\n"
                   "  \"cpu_seconds_load\": %.4f,\n  \"cpu_seconds_tokenize\": %.4f,\n  \"cpu_seconds_invert\": %.4f,\n"
                   "  \"cpu_seconds_write\": %.4f,\n  \"load_avg_1m\": %.2f\n}\n",
                   cli::json_escape(corpus).c_str(), opts.stem ? "true" : "false",
                   opts.stopwords ? "true" : "false", threads, st.num_docs, st.num_terms,
                   (unsigned long long)st.num_postings, (unsigned long long)st.num_positions,
                   ix.avg_doc_len(), (unsigned long long)st.file_bytes,
                   (unsigned long long)st.doc_stream_bytes, (unsigned long long)st.pos_stream_bytes,
                   (unsigned long long)st.raw_doc_stream_bytes,
                   (unsigned long long)st.raw_pos_stream_bytes,
                   (unsigned long long)(st.section_bytes[se::S_LEX_ENTRIES] + st.section_bytes[se::S_LEX_STRINGS]),
                   (unsigned long long)(st.section_bytes[se::S_DOC_LENS] + st.section_bytes[se::S_DOCID_OFFS] +
                                        st.section_bytes[se::S_DOCID_BLOB]),
                   ratio, 8.0 * st.doc_stream_bytes / std::max<uint64_t>(1, st.num_postings),
                   8.0 * st.pos_stream_bytes / std::max<uint64_t>(1, st.num_positions), t_load, t_tok,
                   t_inv, t_write, total, t_open, c_load, c_tok, c_inv, c_write, load0);
      std::fclose(f);
    }
  } catch (const std::exception& e) {
    cli::die(e.what());
  }
  return 0;
}
