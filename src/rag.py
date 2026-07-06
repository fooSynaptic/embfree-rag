"""Emb-free RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import EmbeddingConfig, RAGConfig, config, embedding_config
from src.index import CorpusIndex, IndexBuilder
from src.retrieval import HybridRetriever, build_neural_backend
from src.synthesizer import ExtractiveSynthesizer, RAGAnswer


@dataclass
class EmbFreeRAG:
    """RAG without mandatory neural embeddings."""

    rag_config: RAGConfig = config
    embed_config: EmbeddingConfig = embedding_config

    def __post_init__(self) -> None:
        neural = build_neural_backend(self.embed_config)
        self.index_builder = IndexBuilder(self.rag_config)
        self.retriever = HybridRetriever(self.rag_config, self.embed_config, neural)
        self.synthesizer = ExtractiveSynthesizer()

    def index(self, passage: str) -> CorpusIndex:
        return self.index_builder.build_from_text(passage)

    def query(self, passage: str, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        corpus_index = self.index(passage)
        retrieved = self.retriever.retrieve(corpus_index, question, top_k=top_k)
        return self.synthesizer.synthesize(
            question,
            retrieved,
            corpus_index,
            continuity_window=self.rag_config.continuity_window,
        )

    def format_response(self, result: RAGAnswer) -> str:
        topic_block = "\n".join(
            f"Topic #{topic_id}: {' '.join(words[:10])}"
            for topic_id, words in sorted(result.topics.items())
        )
        return (
            f"Mode: emb-free lexical RAG\n"
            f"Avg score: {result.metrics.get('avg_score', 0):.3f} | "
            f"Continuity: {result.metrics.get('continuity', 0):.3f}\n\n"
            f"{topic_block}\n\n"
            f"{result.answer}\n\n"
            f"--- Retrieved context ---\n{result.context}"
        )


def ask(passage: str, question: str) -> str:
    """Run emb-free RAG and return a formatted response string."""
    pipeline = EmbFreeRAG()
    result = pipeline.query(passage, question)
    return pipeline.format_response(result)
