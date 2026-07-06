"""Optional neural embedding backend."""

from __future__ import annotations

from typing import List

import numpy as np

from src.embeddings.base import EmbeddingBackend


class SentenceTransformerBackend(EmbeddingBackend):
    """Optional open-source embedding model (not required for emb-free mode)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._corpus_cache: List[str] = []

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install optional deps: pip install -r requirements-embeddings.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: List[str]) -> None:
        self._corpus_cache = list(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        return np.asarray(model.encode(texts, normalize_embeddings=True))
