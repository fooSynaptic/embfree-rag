"""Emb-free lexical vectors via TF-IDF."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings.base import EmbeddingBackend


class LexicalEmbeddingBackend(EmbeddingBackend):
    """TF-IDF vectors — default emb-free retriever."""

    def __init__(self, stop_words: Optional[List[str]] = None, **vectorizer_kwargs) -> None:
        self._vectorizer = TfidfVectorizer(stop_words=stop_words, **vectorizer_kwargs)
        self._fitted = False

    @property
    def name(self) -> str:
        return "lexical-tfidf"

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer

    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray()

    def similarity(self, query: str, corpus_matrix: np.ndarray, corpus_texts: List[str]) -> np.ndarray:
        query_vec = self._vectorizer.transform([query]).toarray()
        corpus_vecs = corpus_matrix if corpus_matrix.size else self.encode(corpus_texts)
        return cosine_similarity(query_vec, corpus_vecs).flatten()
