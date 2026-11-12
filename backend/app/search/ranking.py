"""Ranking: BM25 over chunks, cosine similarity over embeddings, and a hybrid of both.

BM25 needs no model and almost no memory, so search works at every effort level and even when the LLM
runtime is down. When embeddings exist (an embedding model is configured), the two scores are blended so
results also match by meaning ("login broken" finds "users can't sign in").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were will with this "
    "these those not no do does did i you we they he she our your their my me".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in (m.group(0).lower() for m in _WORD.finditer(text or "")) if w not in _STOP and len(w) > 1]


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for counts, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                f = counts.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def hybrid(lexical: list[float], semantic: list[float] | None, weight: float = 0.5) -> list[float]:
    """Blend normalised BM25 with cosine similarity (both mapped to 0..1)."""
    top = max(lexical) if lexical else 0.0
    lex = [s / top if top else 0.0 for s in lexical]
    if semantic is None:
        return lex
    sem = [max(0.0, s) for s in semantic]
    return [(1 - weight) * l + weight * s for l, s in zip(lex, sem)]
