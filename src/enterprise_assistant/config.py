"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Public corpus: GitLab's product documentation (CC BY-SA 4.0), pinned to one commit so the
# benchmark is reproducible. Only these sections are ingested.
CORPUS_REPO = "https://github.com/gitlabhq/gitlabhq.git"
CORPUS_COMMIT = "d7b9a6176ab4e279465ba3ee09922cde2b74de47"
CORPUS_PATHS = [
    "doc/ci",
    "doc/user/project/merge_requests",
    "doc/user/project/repository",
    "doc/user/project/issues",
    "doc/user/packages",
    "doc/administration/backup_restore",
    "doc/user/profile",
    "doc/user/group",
    "doc/topics/git",
]
DOCS_BASE_URL = "https://docs.gitlab.com"

# USD per million tokens, from https://platform.claude.com/docs/en/about-claude/pricing
# and https://developers.openai.com/api/docs/models/gpt-5.4-mini (checked 2026-10-27). Used for cost-per-query reporting.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "offline-extractive": (0.0, 0.0),
}


@dataclass
class Settings:
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "postgresql://postgres@localhost:5432/assistant"
        )
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("ASSISTANT_DATA_DIR", "data")))
    # offline | anthropic | openai
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "offline"))
    anthropic_model: str = field(
        default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    )
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-5.4-mini"))
    # offline (LSA, no network) | openai
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "offline"))
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "768")))
    # Model used to *project* cost per query when running offline.
    cost_reference_model: str = field(
        default_factory=lambda: _env("COST_REFERENCE_MODEL", "claude-sonnet-5-5")
    )
    candidates_k: int = 40  # per retriever, before fusion
    rerank_k: int = 6  # chunks passed to the generator
    # Answer only when the reranker is confident AND the retrieved text covers enough of the
    # question's (IDF-weighted) terms. Both are tuned on the dev split by `assistant tune`.
    min_answer_confidence: float = field(
        default_factory=lambda: float(_env("MIN_ANSWER_CONFIDENCE", "0.02"))
    )
    min_query_coverage: float = field(
        default_factory=lambda: float(_env("MIN_QUERY_COVERAGE", "0.0"))
    )

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "gitlab-docs"


def get_settings() -> Settings:
    return Settings()
