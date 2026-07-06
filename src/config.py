"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / ".cache"


@dataclass(frozen=True)
class RAGConfig:
    """Defaults for lexical / emb-free RAG."""

    stopwords_file: Path = DATA_DIR / "stopwords.txt"
    vector_cache: Path = CACHE_DIR / "vectorizer.pkl"
    max_df: float = 0.95
    min_df: int = 1
    max_features: int = 3000
    n_topics: int = 8
    n_top_words: int = 10
    factor_method: str = "NMF"
    top_k: int = 5
    topic_boost: float = 0.35
    continuity_window: int = 3
    sentence_delimiters: Tuple[str, ...] = ("。", "！", "？", "!", "?")


@dataclass
class EmbeddingConfig:
    """Optional neural embedding backend (disabled by default)."""

    backend: str = "lexical"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    hybrid_alpha: float = 0.7


config = RAGConfig()
embedding_config = EmbeddingConfig()
