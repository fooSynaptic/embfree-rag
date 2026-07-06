"""
Backward-compatible entry point.

The old `paraphrase()` API now routes through emb-free RAG. Provide a
question when possible; otherwise a default topic question is used.
"""

from __future__ import annotations

from src.rag import EmbFreeRAG, ask


def paraphrase(
    texts: str,
    n_components: int = 25,
    mode: str = "input",
    sent_tokenize: str = "。",
    question: str = "",
) -> str:
    del n_components, mode, sent_tokenize
    pipeline = EmbFreeRAG()
    query = question.strip() or "这段文本的主要话题和相关内容是什么？"
    result = pipeline.query(texts, query)
    return pipeline.format_response(result)


__all__ = ["paraphrase", "ask"]
