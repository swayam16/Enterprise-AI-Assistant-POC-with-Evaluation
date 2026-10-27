"""Postgres access: schema, bulk load of chunks, and connection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .corpus import Chunk


def ensure_database(database_url: str) -> None:
    """Create the target database if it does not exist (handy for local runs and CI)."""
    info = psycopg.conninfo.conninfo_to_dict(database_url)
    name = info.get("dbname", "postgres")
    admin = psycopg.conninfo.make_conninfo(database_url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


@contextmanager
def connect(database_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        yield conn


def create_schema(conn: psycopg.Connection, dim: int) -> None:
    sql = resources.files("enterprise_assistant").joinpath("sql/schema.sql").read_text()
    conn.execute(sql.replace("%(dim)s", str(int(dim))))
    conn.commit()


def load_chunks(conn: psycopg.Connection, chunks: list[Chunk], vectors: np.ndarray) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE chunks")
        with cur.copy(
            "COPY chunks (chunk_id, doc_id, url, title, section, text, embedding) FROM STDIN"
        ) as copy:
            for c, v in zip(chunks, vectors, strict=True):
                copy.write_row((c.chunk_id, c.doc_id, c.url, c.title, c.section, c.text, v))
    conn.commit()


def build_indexes(conn: psycopg.Connection) -> None:
    conn.execute("SET maintenance_work_mem = '256MB'")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)")
    conn.execute("ANALYZE chunks")
    conn.commit()
