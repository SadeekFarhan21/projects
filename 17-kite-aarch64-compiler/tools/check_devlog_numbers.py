#!/usr/bin/env python3
"""List every number in DEVLOG.md that does not appear verbatim in some
file under results/. Register names (x0, x19, ...), the Mach-O symbol
prefixes and bit widths written as 2^64 are ignored because they are
names and definitions, not measurements. Exit status 1 if anything
else is missing."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text = open(os.path.join(ROOT, "DEVLOG.md")).read()
corpus = ""
for f in sorted(os.listdir(os.path.join(ROOT, "results"))):
    corpus += open(os.path.join(ROOT, "results", f)).read()
corpus_nocomma = corpus.replace(",", "")

# drop things that are names or definitions, not measurements
cleaned = re.sub(r"\b[xw]\d+\b", " ", text)          # registers
cleaned = re.sub(r"2\^\d+", " ", cleaned)             # 2^64, 2^63
cleaned = re.sub(r"`[^`]*`", " ", cleaned)            # inline code
cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)

missing = []
for m in re.finditer(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", cleaned):
    num = m.group(0).rstrip(",")
    plain = num.replace(",", "")
    if num in corpus or plain in corpus_nocomma:
        continue
    missing.append(num)

if missing:
    print("numbers in DEVLOG.md not found in results/:", ", ".join(sorted(set(missing), key=missing.index)))
    sys.exit(1)
print("every number in DEVLOG.md appears in a results/ file")
