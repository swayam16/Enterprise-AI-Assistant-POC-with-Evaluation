"""Hybrid retrieval inside Postgres: pgvector (HNSW, cosine) + full-text search, fused with
reciprocal rank fusion (RRF) in a single SQL statement."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import psycopg

RRF_K = 60

_STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "use",
        "using",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "gitlab",
    ]
)

HYBRID_SQL = """
WITH dense AS (
    SELECT chunk_id, 1 - (embedding <=> %(qvec)s) AS dense_sim,
           row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS dense_rank
    FROM chunks
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(k)s
),
lexical AS (
    SELECT chunk_id, ts_rank(tsv, q) AS lex_score,
           row_number() OVER (ORDER BY ts_rank(tsv, q) DESC) AS lex_rank
    FROM chunks, to_tsquery('english', %(tsq)s) AS q
    WHERE %(tsq)s <> '' AND tsv @@ q
    ORDER BY lex_score DESC
    LIMIT %(k)s
),
fused AS (
    SELECT coalesce(d.chunk_id, l.chunk_id) AS chunk_id,
           coalesce(d.dense_sim, 0) AS dense_sim, coalesce(d.dense_rank, 0) AS dense_rank,
           coalesce(l.lex_score, 0) AS lex_score, coalesce(l.lex_rank, 0) AS lex_rank,
           coalesce(1.0 / (%(rrf_k)s + d.dense_rank), 0)
             + coalesce(1.0 / (%(rrf_k)s + l.lex_rank), 0) AS rrf
    FROM dense d FULL OUTER JOIN lexical l USING (chunk_id)
)
SELECT f.chunk_id, c.doc_id, c.url, c.title, c.section, c.text,
       f.dense_sim, f.dense_rank, f.lex_score, f.lex_rank, f.rrf
FROM fused f JOIN chunks c USING (chunk_id)
ORDER BY f.rrf DESC
"""


@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    section: str
    text: str
    dense_sim: float
    dense_rank: int
    lex_score: float
    lex_rank: int
    rrf: float
    score: float = 0.0  # set by the reranker

    @property
    def section_key(self) -> str:
        return f"{self.doc_id}#{self.section}"


def query_terms(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.-]*[A-Za-z0-9_]|[A-Za-z0-9]", query.lower())
    return [w for w in words if w not in _STOP and len(w) > 1]


def to_tsquery_text(query: str) -> str:
    """OR together the content words. Postgres stems them; RRF + reranker handle precision."""
    terms = []
    for t in query_terms(query):
        t = re.sub(r"[^a-z0-9_]", " ", t).strip()
        terms.extend(p for p in t.split() if p and p not in _STOP)
    return " | ".join(dict.fromkeys(terms))


def hybrid_search(
    conn: psycopg.Connection, query: str, qvec: np.ndarray, k: int = 40
) -> list[Candidate]:
    conn.execute("SET hnsw.ef_search = 100")
    rows = conn.execute(
        HYBRID_SQL,
        {"qvec": qvec, "k": k, "tsq": to_tsquery_text(query), "rrf_k": RRF_K},
    ).fetchall()
    return [Candidate(*r) for r in rows]


def dense_only(conn: psycopg.Connection, qvec: np.ndarray, k: int = 40) -> list[str]:
    conn.execute("SET hnsw.ef_search = 100")
    rows = conn.execute(
        "SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (qvec, k)
    ).fetchall()
    return [r[0] for r in rows]


def lexical_only(conn: psycopg.Connection, query: str, k: int = 40) -> list[str]:
    tsq = to_tsquery_text(query)
    if not tsq:
        return []
    rows = conn.execute(
        "SELECT chunk_id FROM chunks, to_tsquery('english', %s) q WHERE tsv @@ q "
        "ORDER BY ts_rank(tsv, q) DESC LIMIT %s",
        (tsq, k),
    ).fetchall()
    return [r[0] for r in rows]
