"""Embedding backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingBackend(ABC):
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
