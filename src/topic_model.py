"""Matrix factorization topic modeling (NMF / LDA)."""

from __future__ import annotations

from time import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF


def extract_top_words(model, feature_names: List[str], n_top_words: int) -> Dict[int, List[str]]:
    topics: Dict[int, List[str]] = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        topics[topic_idx] = [feature_names[i] for i in top_indices]
    return topics


def factorize_matrix(
    matrix: np.ndarray,
    n_components: int,
    feature_names: List[str],
    n_top_words: int,
    factor_method: str = "NMF",
) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    if factor_method == "NMF":
        model = NMF(
            n_components=n_components,
            random_state=1,
            beta_loss="kullback-leibler",
            solver="mu",
            max_iter=1000,
        ).fit(matrix)
    elif factor_method == "LDA":
        model = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=10,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        ).fit(matrix)
    else:
        raise ValueError(f"Unknown factor method: {factor_method}")

    topic_words = extract_top_words(model, feature_names, n_top_words)
    doc_topic = model.transform(matrix)
    return doc_topic, topic_words
