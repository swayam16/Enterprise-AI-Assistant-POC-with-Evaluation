"""Dense embeddings.

`offline` is a latent-semantic model (TF-IDF over word unigrams/bigrams, reduced with SVD) fitted
on the corpus itself. It needs no network or API key, so CI and the benchmark are free and
repeatable. `openai` uses text-embedding-3-small truncated to the same dimension, so either
provider fits the same pgvector column.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    name = "offline-lsa"

    def __init__(self, dim: int = 256, pipeline=None):
        self.dim = dim
        self.pipeline = pipeline
        self._prepare()

    def _prepare(self) -> None:
        # sklearn's TruncatedSVD.transform multiplies by a transposed, non-contiguous matrix,
        # which costs ~140 ms per query; see embed() for the fast path.
        if self.pipeline is not None:
            self._tfidf = self.pipeline.steps[0][1]
            self._proj = np.ascontiguousarray(
                self.pipeline.steps[1][1].components_.T, dtype=np.float32
            )

    def fit(self, texts: list[str], seed: int = 7) -> LSAEmbedder:
        self.pipeline = make_pipeline(
            TfidfVectorizer(
                sublinear_tf=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.5,
                max_features=200_000,
                token_pattern=r"(?u)\b[\w.$-]*\w\b",
            ),
            TruncatedSVD(
                n_components=self.dim, algorithm="randomized", n_iter=7, random_state=seed
            ),
            Normalizer(copy=False),
        )
        self.pipeline.fit(texts)
        self._prepare()
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        X = self._tfidf.transform(texts).tocsr()
        dense = np.zeros((X.shape[0], self._proj.shape[1]), dtype=np.float32)
        for i in range(X.shape[0]):  # gather rows: ~0.1 ms/query vs ~45 ms for scipy's sparse@dense
            lo, hi = X.indptr[i], X.indptr[i + 1]
            if hi > lo:
                dense[i] = X.data[lo:hi].astype(np.float32) @ self._proj[X.indices[lo:hi]]
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return dense / np.where(norms == 0, 1, norms)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"dim": self.dim, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: Path) -> LSAEmbedder:
        blob = joblib.load(path)
        return cls(dim=blob["dim"], pipeline=blob["pipeline"])


class OpenAIEmbedder:
    name = "openai-text-embedding-3-small"

    def __init__(self, dim: int = 256, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.dim, self.model = dim, model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            resp = self.client.embeddings.create(
                model=self.model, input=texts[i : i + 256], dimensions=self.dim
            )
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def load_embedder(provider: str, artifacts_dir: Path, dim: int) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(dim=dim)
    return LSAEmbedder.load(artifacts_dir / "lsa_embedder.joblib")
