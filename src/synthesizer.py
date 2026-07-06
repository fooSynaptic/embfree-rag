"""Answer synthesis from retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.index import CorpusIndex
from src.retrieval import RetrievedChunk


@dataclass
class RAGAnswer:
    answer: str
    context: str
    topics: Dict[int, List[str]]
    citations: List[str]
    metrics: Dict[str, float]


class ExtractiveSynthesizer:
    """Build a readable, citation-aware extractive answer."""

    def synthesize(
        self,
        query: str,
        retrieved: Sequence[RetrievedChunk],
        index: CorpusIndex,
        continuity_window: int = 3,
    ) -> RAGAnswer:
        if not retrieved:
            return RAGAnswer(
                answer="未检索到与问题相关的句子。",
                context="",
                topics=index.topic_words,
                citations=[],
                metrics={"avg_score": 0.0, "continuity": 0.0},
            )

        ordered = sorted(retrieved, key=lambda item: item.chunk.chunk_id)
        continuity = self._continuity_rate([item.chunk.chunk_id for item in ordered], continuity_window)
        avg_score = sum(item.score for item in retrieved) / len(retrieved)

        topic_lines = []
        for topic_id, words in sorted(index.topic_words.items()):
            topic_lines.append(f"Topic #{topic_id}: {' '.join(words[:10])}")

        citation_lines = []
        for item in ordered:
            topic_id = item.chunk.topic_id if item.chunk.topic_id is not None else -1
            citation_lines.append(
                f"[{item.chunk.chunk_id}] (topic={topic_id}, score={item.score:.3f}) {item.chunk.text}"
            )

        context = "\n".join(citation_lines)
        answer = self._compose_answer(query, ordered)

        return RAGAnswer(
            answer=answer,
            context=context,
            topics=index.topic_words,
            citations=[line for line in citation_lines],
            metrics={"avg_score": avg_score, "continuity": continuity},
        )

    def _compose_answer(self, query: str, retrieved: Sequence[RetrievedChunk]) -> str:
        lead = f"问题：{query.strip()}\n\n相关原文摘录（按语料顺序排列）：\n"
        body = "\n".join(f"- {item.chunk.text}" for item in retrieved)
        return lead + body

    def _continuity_rate(self, indices: Sequence[int], window: int) -> float:
        if len(indices) < 2:
            return 1.0
        pairs = [indices[i] - indices[i - 1] <= window for i in range(1, len(indices))]
        return sum(pairs) / len(pairs)
