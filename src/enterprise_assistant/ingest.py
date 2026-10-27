"""Ingestion pipeline: fetch docs -> chunk -> embed -> load into Postgres -> build indexes."""

from __future__ import annotations

import json
import math
import time

from . import db
from .config import CORPUS_COMMIT, Settings
from .corpus import Chunk, build_chunks, fetch_docs, save_chunks
from .embeddings import LSAEmbedder, OpenAIEmbedder
from .guardrails import content_tokens


def index_text(c: Chunk) -> str:
    return f"{c.title}. {c.section}. {c.text}"


def write_idf(texts: list[str], path) -> None:
    """IDF of the same stemmed content tokens the generator and guardrails use."""
    df: dict[str, int] = {}
    for t in texts:
        for tok in content_tokens(t):
            df[tok] = df.get(tok, 0) + 1
    n = len(texts)
    idf = {tok: round(math.log((n + 1) / (d + 1)) + 1, 4) for tok, d in df.items()}
    path.write_text(json.dumps(idf))


def run_ingest(settings: Settings, log=print) -> dict:
    t0 = time.perf_counter()
    docs_dir = fetch_docs(settings.docs_dir)
    chunks = build_chunks(docs_dir)
    save_chunks(chunks, settings.artifacts_dir / "chunks.jsonl")
    log(f"chunked {len({c.doc_id for c in chunks})} pages into {len(chunks)} chunks")

    texts = [index_text(c) for c in chunks]
    write_idf(texts, settings.artifacts_dir / "idf.json")
    if settings.embedding_provider == "openai":
        embedder = OpenAIEmbedder(dim=settings.embedding_dim)
    else:
        embedder = LSAEmbedder(dim=settings.embedding_dim).fit(texts)
        embedder.save(settings.artifacts_dir / "lsa_embedder.joblib")
    vectors = embedder.embed(texts)
    log(f"embedded with {embedder.name} ({vectors.shape[1]} dims)")

    db.ensure_database(settings.database_url)
    with db.connect(settings.database_url) as conn:
        db.create_schema(conn, vectors.shape[1])
        db.load_chunks(conn, chunks, vectors)
        db.build_indexes(conn)
    stats = {
        "corpus_commit": CORPUS_COMMIT,
        "pages": len({c.doc_id for c in chunks}),
        "chunks": len(chunks),
        "words": sum(len(c.text.split()) for c in chunks),
        "embedder": embedder.name,
        "dim": int(vectors.shape[1]),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (settings.artifacts_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"loaded into Postgres with HNSW + GIN indexes in {stats['seconds']}s")
    return stats
