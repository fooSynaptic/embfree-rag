"""Retrieval: lexical + topic routing (+ optional hybrid embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EmbeddingConfig, RAGConfig, config, embedding_config
from src.embeddings.base import EmbeddingBackend
from src.embeddings.lexical import LexicalEmbeddingBackend
from src.embeddings.sentence_transformer import SentenceTransformerBackend
from src.index import Chunk, CorpusIndex
from src.preprocessing import tokenize


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    lexical_score: float
    topic_score: float
    neural_score: float = 0.0


class HybridRetriever:
    """Emb-free retriever with optional neural reranking."""

    def __init__(
        self,
        rag_config: RAGConfig = config,
        embed_config: EmbeddingConfig = embedding_config,
        neural_backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.rag_config = rag_config
        self.embed_config = embed_config
        self.neural_backend = neural_backend

    def retrieve(self, index: CorpusIndex, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if index.size == 0:
            return []

        top_k = top_k or self.rag_config.top_k
        query_tokens = tokenize(query)
        lexical_scores = self._lexical_scores(index, query_tokens)
        topic_scores = self._topic_scores(index, query_tokens)
        neural_scores = self._neural_scores(index, query)

        alpha = self.embed_config.hybrid_alpha
        if self.neural_backend is None:
            combined = (1.0 - self.rag_config.topic_boost) * lexical_scores + self.rag_config.topic_boost * topic_scores
        else:
            combined = (
                alpha * lexical_scores
                + (1.0 - alpha) * neural_scores
                + self.rag_config.topic_boost * topic_scores
            )

        ranked_indices = np.argsort(combined)[::-1][:top_k]
        results: List[RetrievedChunk] = []
        for idx in ranked_indices:
            results.append(
                RetrievedChunk(
                    chunk=index.chunks[int(idx)],
                    score=float(combined[idx]),
                    lexical_score=float(lexical_scores[idx]),
                    topic_score=float(topic_scores[idx]),
                    neural_score=float(neural_scores[idx]),
                )
            )
        return results

    def _lexical_scores(self, index: CorpusIndex, query_tokens: str) -> np.ndarray:
        backend = index.backend
        if not isinstance(backend, LexicalEmbeddingBackend):
            raise TypeError("Corpus index must use LexicalEmbeddingBackend for emb-free mode")
        return backend.similarity(query_tokens, index.matrix, [chunk.tokens for chunk in index.chunks])

    def _topic_scores(self, index: CorpusIndex, query_tokens: str) -> np.ndarray:
        query_words = set(query_tokens.split())
        scores = np.zeros(index.size, dtype=float)
        for chunk in index.chunks:
            topic_id = chunk.topic_id or 0
            topic_words = index.topic_words.get(topic_id, [])
            if not topic_words:
                continue
            overlap = len(query_words.intersection(topic_words))
            scores[chunk.chunk_id] = overlap / len(topic_words)
        return scores

    def _neural_scores(self, index: CorpusIndex, query: str) -> np.ndarray:
        if self.neural_backend is None:
            return np.zeros(index.size, dtype=float)
        corpus_texts = [chunk.text for chunk in index.chunks]
        self.neural_backend.fit(corpus_texts)
        query_vec = self.neural_backend.encode([query])
        corpus_vecs = self.neural_backend.encode(corpus_texts)
        return cosine_similarity(query_vec, corpus_vecs).flatten()


def build_neural_backend(embed_config: EmbeddingConfig = embedding_config) -> Optional[EmbeddingBackend]:
    if embed_config.backend == "lexical":
        return None
    if embed_config.backend == "sentence-transformers":
        return SentenceTransformerBackend(embed_config.model_name)
    raise ValueError(f"Unknown embedding backend: {embed_config.backend}")
