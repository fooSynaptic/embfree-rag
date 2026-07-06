"""Corpus indexing for emb-free RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.config import RAGConfig, config
from src.embeddings.base import EmbeddingBackend
from src.embeddings.lexical import LexicalEmbeddingBackend
from src.preprocessing import load_stopwords, preprocess_corpus, split_sentences
from src.topic_model import factorize_matrix


@dataclass
class Chunk:
    chunk_id: int
    text: str
    tokens: str
    topic_id: Optional[int] = None


@dataclass
class CorpusIndex:
    chunks: List[Chunk] = field(default_factory=list)
    matrix: Optional[np.ndarray] = None
    doc_topic: Optional[np.ndarray] = None
    topic_words: Dict[int, List[str]] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    backend: Optional[EmbeddingBackend] = None

    @property
    def size(self) -> int:
        return len(self.chunks)


class IndexBuilder:
    """Build a lexical index with optional topic decomposition."""

    def __init__(self, rag_config: RAGConfig = config, backend: Optional[EmbeddingBackend] = None) -> None:
        self.config = rag_config
        stopwords = load_stopwords(rag_config.stopwords_file)
        self.backend = backend or LexicalEmbeddingBackend(
            stop_words=stopwords,
            max_df=rag_config.max_df,
            min_df=rag_config.min_df,
            max_features=rag_config.max_features,
        )

    def build_from_text(self, text: str) -> CorpusIndex:
        raw_sentences = split_sentences(text, self.config.sentence_delimiters)
        return self.build_from_sentences(raw_sentences)

    def build_from_sentences(self, raw_sentences: Sequence[str]) -> CorpusIndex:
        tokenized = preprocess_corpus(raw_sentences)
        chunks = [
            Chunk(chunk_id=idx, text=raw_sentences[idx], tokens=tokenized[idx])
            for idx in range(len(raw_sentences))
        ]

        self.backend.fit(tokenized)
        matrix = self.backend.encode(tokenized)

        n_components = min(self.config.n_topics, max(2, len(chunks) // 3))
        doc_topic, topic_words = factorize_matrix(
            matrix,
            n_components=n_components,
            feature_names=list(self.backend.vectorizer.get_feature_names_out()),
            n_top_words=self.config.n_top_words,
            factor_method=self.config.factor_method,
        )

        for idx, chunk in enumerate(chunks):
            chunk.topic_id = int(np.argmax(doc_topic[idx]))

        return CorpusIndex(
            chunks=chunks,
            matrix=matrix,
            doc_topic=doc_topic,
            topic_words=topic_words,
            feature_names=list(self.backend.vectorizer.get_feature_names_out()),
            backend=self.backend,
        )
