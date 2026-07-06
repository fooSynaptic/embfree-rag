"""Embedding backends: lexical (default) or optional sentence-transformers."""

from src.embeddings.base import EmbeddingBackend
from src.embeddings.lexical import LexicalEmbeddingBackend
from src.embeddings.sentence_transformer import SentenceTransformerBackend

__all__ = [
    "EmbeddingBackend",
    "LexicalEmbeddingBackend",
    "SentenceTransformerBackend",
]
