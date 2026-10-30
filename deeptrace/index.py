"""Sentence-embedding index over reference passages, searched with FAISS."""
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDER = os.getenv(
    "DEEPTRACE_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"
)


def load_embedder(model_name: str = DEFAULT_EMBEDDER) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class PassageIndex:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.index = None
        self.passages: list[str] = []
        self.sources: list[str] = []

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.astype("float32")

    def add(self, passages: list[str], sources: list[str]) -> None:
        if len(passages) != len(sources):
            raise ValueError("Passages and sources must have the same length.")
        if not passages:
            return
        vectors = self._encode(passages)
        if self.index is None:
            # Inner product on normalized vectors equals cosine similarity.
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.passages.extend(passages)
        self.sources.extend(sources)

    def search(self, queries: list[str], k: int = 3):
        """For each query, return a list of (score, passage, source) hits."""
        if self.index is None or not queries:
            return [[] for _ in queries]
        scores, ids = self.index.search(self._encode(queries), min(k, self.index.ntotal))
        results = []
        for row_scores, row_ids in zip(scores, ids):
            hits = [
                (float(s), self.passages[i], self.sources[i])
                for s, i in zip(row_scores, row_ids)
                if i != -1
            ]
            results.append(hits)
        return results
