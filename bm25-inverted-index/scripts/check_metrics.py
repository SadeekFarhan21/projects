# /// script
# requires-python = ">=3.10"
# dependencies = ["pytrec-eval-terrier>=0.5.6"]
# ///
"""Recompute nDCG@10, MRR@10 and Recall@100 for a TREC run file with
pytrec_eval (the library BEIR itself uses) and compare against se_eval's JSON.

usage: uv run scripts/check_metrics.py run.trec qrels.tsv se_eval.json [--ignore-identical-ids]
"""
import json
import sys
from collections import defaultdict

import pytrec_eval

run_path, qrels_path, ours_path = sys.argv[1:4]
ignore_ids = "--ignore-identical-ids" in sys.argv

qrels = defaultdict(dict)
with open(qrels_path) as f:
    next(f)  # header
    for line in f:
        q, d, s = line.rstrip("\n").split("\t")
        if int(s) > 0:
            qrels[q][d] = int(s)

# Keep our rank order exactly: trec_eval re-sorts by score and breaks ties by
# doc id, so feed it strictly decreasing pseudo-scores derived from the rank.
run = defaultdict(dict)
with open(run_path) as f:
    for line in f:
        q, _, d, rank, _score, _ = line.split()
        if ignore_ids and q == d:
            continue
        run[q][d] = 1000.0 - int(rank)

ev = pytrec_eval.RelevanceEvaluator(dict(qrels), {"ndcg_cut.10", "recip_rank", "recall.100"})
per_q = ev.evaluate(dict(run))
qs = [q for q in qrels]  # queries with judgments; a query with no hits scores 0
ndcg = sum(per_q.get(q, {}).get("ndcg_cut_10", 0.0) for q in qs) / len(qs)
rec = sum(per_q.get(q, {}).get("recall_100", 0.0) for q in qs) / len(qs)

# pytrec_eval's recip_rank has no cut-off; compute MRR@10 from the run directly.
def mrr10(q):
    ranked = sorted(run.get(q, {}).items(), key=lambda kv: -kv[1])[:10]
    for i, (d, _) in enumerate(ranked):
        if qrels[q].get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

mrr = sum(mrr10(q) for q in qs) / len(qs)
ours = json.load(open(ours_path))
out = {
    "queries": len(qs),
    "pytrec_eval_ndcg@10": ndcg, "se_eval_ndcg@10": ours["ndcg@10"],
    "python_mrr@10": mrr, "se_eval_mrr@10": ours["mrr@10"],
    "pytrec_eval_recall@100": rec, "se_eval_recall@100": ours["recall@100"],
    "max_abs_diff": max(abs(ndcg - ours["ndcg@10"]), abs(mrr - ours["mrr@10"]), abs(rec - ours["recall@100"])),
}
print(json.dumps(out, indent=2))
