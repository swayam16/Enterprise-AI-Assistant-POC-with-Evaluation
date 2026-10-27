"""Second-stage reranker over the hybrid candidates.

A small logistic-regression model scores each (query, chunk) pair from retrieval signals
(dense similarity and rank, full-text score and rank, RRF) plus overlap features (query-term
coverage in the body and in the title/heading, matching bigrams, exact identifier matches).

It is trained with *synthetic* queries generated from the corpus itself (section headings and
noisy sentence fragments), with every benchmark gold section held out, so the benchmark never
leaks into training. Its calibrated probability for the top chunk doubles as the answer
confidence used to decline questions the docs do not cover.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .guardrails import content_tokens
from .llm import weighted_coverage
from .search import Candidate

FEATURES = [
    "dense_sim",
    "dense_recip",
    "lex_score",
    "lex_recip",
    "rrf",
    "both_lists",
    "body_cov",
    "head_cov",
    "best_sentence_cov",
    "bigrams",
    "ident_match",
    "log_len",
    "troubleshooting_match",
]
_SENT = re.compile(r"(?<=[.!?:])\s+|\n+")
_IDENT = re.compile(r"`([^`]+)`|\b([A-Za-z_]+[_.:/][A-Za-z0-9_.:/]+|[A-Z][A-Z0-9_]{3,})\b")
_ERRORISH = re.compile(
    r"\b(error|fail|fails|failed|failing|forbidden|denied|not working|"
    r"timeout|timing out|can't|cannot|unable|missing|broken|stuck)\b",
    re.I,
)


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


def _ordered_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2]


def features(query: str, c: Candidate) -> list[float]:
    q = content_tokens(query)
    body = content_tokens(c.text)
    head = content_tokens(f"{c.title} {c.section}")
    qb = _bigrams(_ordered_tokens(query))
    cb = _bigrams(_ordered_tokens(f"{c.title} {c.section} {c.text}"))
    idents = {(a or b).lower() for a, b in _IDENT.findall(query)}
    hay = f"{c.title} {c.section} {c.text}".lower()
    ts = "troubleshoot" in (c.title + c.section).lower()
    best_sent = max(
        (
            weighted_coverage(q, content_tokens(s))
            for s in _SENT.split(c.text)
            if len(s.split()) > 3
        ),
        default=0.0,
    )
    dense_sim = c.dense_sim if c.dense_sim == c.dense_sim else 0.0  # NaN for an all-zero query
    return [
        dense_sim,
        1.0 / c.dense_rank if c.dense_rank else 0.0,
        c.lex_score,
        1.0 / c.lex_rank if c.lex_rank else 0.0,
        c.rrf * 30,
        float(bool(c.dense_rank and c.lex_rank)),
        weighted_coverage(q, body),
        weighted_coverage(q, head),
        best_sent,
        min(len(qb & cb), 6) / 6,
        (sum(i in hay for i in idents) / len(idents)) if idents else 0.0,
        math.log1p(len(c.text.split())) / 6,
        float(ts and bool(_ERRORISH.search(query))),
    ]


class Reranker:
    def __init__(self, model=None):
        self.model = model

    def score(self, query: str, cands: list[Candidate]) -> list[Candidate]:
        if not cands:
            return []
        if self.model is None:  # untrained fallback: RRF order
            for c in cands:
                c.score = c.rrf
        else:
            X = np.array([features(query, c) for c in cands])
            for c, p in zip(cands, self.model.predict_proba(X)[:, 1], strict=True):
                c.score = float(p)
        return sorted(cands, key=lambda c: c.score, reverse=True)

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> Reranker:
        return cls(joblib.load(path) if path.exists() else None)


# --- Synthetic training data ------------------------------------------------------------------

_TEMPLATES = ["how do i {}", "{}", "what is {}", "how to {}", "{} in gitlab", "help with {}"]


def synthetic_queries(
    sections: dict[str, tuple[str, str, str]], exclude: set[str], n: int, seed: int = 13
) -> list[tuple[str, str]]:
    """(query, gold section_key) pairs from headings and noisy body fragments."""
    rng = random.Random(seed)
    keys = sorted(
        k
        for k, (_, sec, body) in sections.items()
        if k not in exclude and sec and len(body.split()) > 30
    )
    rng.shuffle(keys)
    out = []
    for key in keys[:n]:
        title, sec, body = sections[key]
        if rng.random() < 0.5:
            leaf = re.sub(r"[`*]", "", sec.split(" > ")[-1]).lower()
            out.append((rng.choice(_TEMPLATES).format(leaf), key))
        else:
            words = re.sub(r"[`*#|]", " ", body).split()
            start = rng.randrange(0, max(1, len(words) - 14))
            frag = [w for w in words[start : start + 14] if rng.random() > 0.35]
            out.append((" ".join(frag), key))
    return out


def train(pairs: list[tuple[list[float], int]], seed: int = 0) -> Reranker:
    X = np.array([f for f, _ in pairs])
    y = np.array([lbl for _, lbl in pairs])
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    )
    model.fit(X, y)
    return Reranker(model)
