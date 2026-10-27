"""Train the reranker on synthetic queries (benchmark gold sections are held out)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import db
from .config import Settings
from .corpus import load_chunks
from .embeddings import load_embedder
from .evaluation.benchmark import load as load_benchmark
from .llm import set_idf
from .rerank import features, synthetic_queries, train
from .search import hybrid_search


def train_reranker(
    settings: Settings, benchmark_path: Path, n_queries: int = 2500, log=print
) -> dict:
    t0 = time.perf_counter()
    set_idf(json.loads((settings.artifacts_dir / "idf.json").read_text()))
    chunks = load_chunks(settings.artifacts_dir / "chunks.jsonl")
    sections: dict[str, tuple[str, str, str]] = {}
    for c in chunks:
        title, sec, body = sections.get(c.section_key, (c.title, c.section, ""))
        sections[c.section_key] = (title, sec, (body + " " + c.text).strip())
    # Every evaluation gold section (benchmark + held-out set) is excluded from training.
    eval_files = [benchmark_path, benchmark_path.parent / "heldout.jsonl"]
    held_out = {
        g
        for path in eval_files
        if path.exists()
        for it in load_benchmark(path)
        for g in it.get("gold_sections", [])
    }
    pairs = synthetic_queries(sections, exclude=held_out, n=n_queries)
    embedder = load_embedder(
        settings.embedding_provider, settings.artifacts_dir, settings.embedding_dim
    )
    vecs = embedder.embed([q for q, _ in pairs])
    rows, with_pos = [], 0
    with db.connect(settings.database_url) as conn:
        for (query, key), v in zip(pairs, vecs, strict=True):
            cands = hybrid_search(conn, query, v, k=settings.candidates_k)
            labels = [int(c.section_key == key) for c in cands]
            with_pos += any(labels)
            rows += [(features(query, c), y) for c, y in zip(cands, labels, strict=True)]
    model = train(rows)
    model.save(settings.artifacts_dir / "reranker.joblib")
    stats = {
        "synthetic_queries": len(pairs),
        "queries_with_gold_in_candidates": with_pos,
        "training_pairs": len(rows),
        "held_out_sections": len(held_out),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (settings.artifacts_dir / "reranker_stats.json").write_text(json.dumps(stats, indent=2))
    log(
        f"reranker trained on {len(rows)} pairs from {len(pairs)} synthetic queries "
        f"({len(held_out)} benchmark sections held out)"
    )
    return stats
