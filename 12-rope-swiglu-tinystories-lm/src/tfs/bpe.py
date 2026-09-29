"""Byte-level BPE tokenizer, trained and applied without any tokenizer library.

Pipeline for encode:
  text -> split on special tokens -> regex pre-tokenize each piece into chunks
       -> UTF-8 bytes of each chunk -> apply learned merges by rank -> ids

Because the base alphabet is all 256 byte values, every string is encodable and
decode(encode(s)) == s holds exactly for any valid str. Merges never cross a
pre-token chunk boundary, which keeps words and their leading spaces together
the way GPT-2 does.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

# GPT-2's pre-tokenizer uses \p{L} and \p{N} from the third-party `regex`
# module. The stdlib `re` has no Unicode property classes, so letters are
# spelled [^\W\d_] ("word chars minus digits and underscore"). That class
# excludes "_", and \d excludes it too, so the punctuation branch must add it
# back explicitly or underscores are silently dropped (see DEVLOG, Problems).
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)

EOT = "<|endoftext|>"


def pretokenize(text: str) -> list[str]:
    return PRETOKEN_PATTERN.findall(text)


def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, scanning left to right."""
    out: list[int] = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: list[str] | None = None):
        self.merges = [tuple(m) for m in merges]
        # rank of a pair == its merge index; lower rank merges first
        self.ranks: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.special_tokens: dict[str, int] = {}
        for tok in special_tokens or []:
            self.special_tokens[tok] = len(self.vocab) + len(self.special_tokens)
        self.special_ids = {v: k for k, v in self.special_tokens.items()}
        self._special_split = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> "BPETokenizer":
        special_tokens = special_tokens or []
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError("vocab_size too small for 256 bytes plus special tokens")
        splitter = (
            re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None
        )

        # 1. Count pre-token chunks. BPE only needs unique chunks and their counts.
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            pieces = splitter.split(text) if splitter else [text]
            for piece in pieces:
                chunk_counts.update(pretokenize(piece))
        words: list[list[int]] = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs: list[int] = list(chunk_counts.values())

        # 2. Global pair counts plus an inverted index pair -> words containing it,
        #    so each merge only touches the words it actually changes.
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(wi)

        # 3. Max-heap with lazy invalidation: entries can be stale, so we check the
        #    popped count against the live count. Ties break on the smaller pair,
        #    which makes training deterministic.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < num_merges and heap:
            neg, pair = heapq.heappop(heap)
            live = pair_counts.get(pair, 0)
            if live <= 0:
                continue
            if -neg != live:  # stale entry; re-push with the live count
                heapq.heappush(heap, (-live, pair))
                continue
            new_id = 256 + len(merges)
            merges.append(pair)
            touched: set[tuple[int, int]] = set()
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                new_w = _merge_ids(w, pair, new_id)
                words[wi] = new_w
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    touched.add(p)
            del pair_counts[pair]
            del where[pair]
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
            if verbose and len(merges) % 500 == 0:
                print(f"  merges: {len(merges)}/{num_merges}")
        return cls(merges, special_tokens)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            # pick the adjacent pair that was learned earliest
            best = min(zip(ids, ids[1:]), key=lambda p: ranks.get(p, 1 << 30))
            rank = ranks.get(best)
            if rank is None:
                break
            ids = _merge_ids(ids, best, 256 + rank)
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode text. With allow_special, literal special-token strings map to their id."""
        if not allow_special or self._special_split is None:
            return self._encode_ordinary(text)
        out: list[int] = []
        for piece in self._special_split.split(text):
            if piece in self.special_tokens:
                out.append(self.special_tokens[piece])
            elif piece:
                out.extend(self._encode_ordinary(piece))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.special_ids:
                parts.append(self.special_ids[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts)

    def decode(self, ids: Iterable[int]) -> str:
        # A generated sequence can end mid-character; "replace" keeps decode total.
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        payload = {
            "type": "byte-bpe",
            "pattern": PRETOKEN_PATTERN.pattern,
            "merges": [list(m) for m in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text())
        if payload.get("pattern") != PRETOKEN_PATTERN.pattern:
            raise ValueError("tokenizer was trained with a different pre-tokenizer pattern")
        return cls([tuple(m) for m in payload["merges"]], payload["special_tokens"])
