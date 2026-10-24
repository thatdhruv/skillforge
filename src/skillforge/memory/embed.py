"""Deterministic offline embedder (feature hashing of words and character trigrams, L2-normalised).

It stands in for a neural embedding model so the stack runs with no downloads. Anything implementing
``embed(text) -> list[float]`` with a fixed ``dim`` can replace it; the memory service only stores vectors.
"""
from __future__ import annotations

import hashlib
import math
import re

DIM = 256


class HashEmbedder:
    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _bucket(self, tok: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = re.findall(r"[a-z0-9_]+", text.lower())
        feats = [(t, 1.0) for t in toks]
        for t in toks:
            p = f"#{t}#"
            feats += [(p[i:i + 3], 0.35) for i in range(len(p) - 2)]
        for tok, w in feats:
            i, sgn = self._bucket(tok)
            v[i] += sgn * w
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
