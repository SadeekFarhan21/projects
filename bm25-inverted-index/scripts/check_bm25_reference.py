# /// script
# requires-python = ">=3.10"
# dependencies = ["bm25s>=0.2", "nltk>=3.8", "numpy", "scipy"]
# ///
"""Independent BM25 reference: rebuild the same ranking with the bm25s library
(method="lucene", same k1/b) on text tokenized in Python with the same rules
as our C++ tokenizer, then compare top-10 lists with our TREC run.

usage: uv run scripts/check_bm25_reference.py data/scifact results/runs/scifact_test.trec results/bm25_reference_scifact.json
"""
import json
import re
import sys
from collections import defaultdict

import bm25s
import numpy as np
from nltk.stem.porter import PorterStemmer

data_dir, run_path, out_path = sys.argv[1:4]
STOP = set("a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with".split())
ps = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)
cache = {}

def tok(s):
    out = []
    for w in re.findall(rb"[A-Za-z0-9\x80-\xff]+", s.encode()):
        if len(w) > 64:
            continue
        w = w.decode("utf-8", "surrogateescape").lower() if w.isascii() else w.lower().decode("utf-8", "surrogateescape")
        if w in STOP:
            continue
        if w.isascii() and w.isalpha():
            if w not in cache:
                cache[w] = ps.stem(w, to_lowercase=False) if len(w) > 2 else w
            w = cache[w]
        out.append(w)
    return out

ids, docs = [], []
for line in open(f"{data_dir}/corpus.jsonl"):
    d = json.loads(line)
    ids.append(d["_id"])
    docs.append(tok(d.get("title", "")) + tok(d["text"]))
qrels = defaultdict(dict)
with open(f"{data_dir}/qrels/test.tsv") as f:
    next(f)
    for line in f:
        q, d, s = line.split("\t")
        qrels[q][d] = int(s)
queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(f"{data_dir}/queries.jsonl")}

r = bm25s.BM25(method="lucene", k1=0.9, b=0.4)
r.index(docs, show_progress=False)
ours = defaultdict(list)
for line in open(run_path):
    q, _, d, rank, score, _ = line.split()
    ours[q].append((d, float(score)))

same_top10 = 0
same_top1 = 0
score_diffs = []
for q in qrels:
    qt = tok(queries[q])
    s = r.get_scores(qt)
    top = np.argsort(-s, kind="stable")[:10]
    ref = [ids[i] for i in top if s[i] > 0]
    mine = [d for d, _ in ours[q][:10]]
    same_top10 += set(ref) == set(mine)
    same_top1 += bool(ref and mine and ref[0] == mine[0])
    for d, sc in ours[q][:10]:
        # Lucene (and bm25s "lucene") drop the constant (k1 + 1) numerator
        # factor, which cannot change the ranking; we keep it, so rescale.
        score_diffs.append(abs(sc / (1 + 0.9) - float(s[ids.index(d)])))
out = {"queries": len(qrels), "identical_top10_sets": same_top10, "identical_top1": same_top1,
       "max_abs_score_diff_top10": max(score_diffs), "mean_abs_score_diff_top10": float(np.mean(score_diffs))}
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps(out, indent=2))
