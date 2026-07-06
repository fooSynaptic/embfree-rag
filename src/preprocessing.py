"""Text preprocessing for Chinese corpora."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence

from jieba import cut


def load_stopwords(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_sentences(text: str, delimiters: Sequence[str]) -> List[str]:
    if not text.strip():
        return []
    pattern = "|".join(re.escape(item) for item in delimiters)
    parts = re.split(pattern, text)
    return [part.strip() for part in parts if part.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"[0-9a-zA-Z]+", " ", text).strip()


def tokenize(text: str) -> str:
    return " ".join(cut(normalize_text(text)))


def preprocess_corpus(sentences: Iterable[str]) -> List[str]:
    return [tokenize(sentence) for sentence in sentences if sentence.strip()]
