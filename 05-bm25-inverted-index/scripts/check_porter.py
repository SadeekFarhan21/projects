# /// script
# requires-python = ">=3.10"
# dependencies = ["nltk>=3.8"]
# ///
"""Diff our C++ Porter stemmer against NLTK's MARTIN_EXTENSIONS mode (the
mode that follows Martin Porter's reference C code) on a real vocabulary.

usage: uv run scripts/check_porter.py build/release/se_stem data/scifact/corpus.jsonl results/porter_check.json
"""
import json
import re
import subprocess
import sys

from nltk.stem.porter import PorterStemmer

se_stem, corpus, out = sys.argv[1], sys.argv[2], sys.argv[3]
vocab = set()
with open(corpus) as f:
    for line in f:
        d = json.loads(line)
        for w in re.findall(r"[a-z]+", (d.get("title", "") + " " + d["text"]).lower()):
            if len(w) <= 64:
                vocab.add(w)
words = sorted(vocab)
res = subprocess.run([se_stem], input="\n".join(words) + "\n", capture_output=True, text=True, check=True)
ours = dict(line.split("\t") for line in res.stdout.splitlines())
ps = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)
diffs = []
for w in words:
    ref = ps.stem(w, to_lowercase=False) if len(w) > 2 else w
    if ours[w] != ref:
        diffs.append({"word": w, "ours": ours[w], "nltk": ref})
summary = {"vocabulary": len(words), "mismatches": len(diffs), "agreement": 1 - len(diffs) / len(words),
           "examples": diffs[:50]}
json.dump(summary, open(out, "w"), indent=2)
print(f"{len(words)} words, {len(diffs)} mismatches ({100 * summary['agreement']:.3f}% agree)")
for d in diffs[:15]:
    print(d)
