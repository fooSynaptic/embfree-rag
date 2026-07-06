#!/usr/bin/env python3
"""CLI demo for emb-free RAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag import EmbFreeRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Emb-free RAG demo")
    parser.add_argument("--passage-file", type=Path, default=ROOT / "data" / "sample_passage.txt")
    parser.add_argument("--question", default="这段对话主要在讨论什么？")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    passage = args.passage_file.read_text(encoding="utf-8")
    pipeline = EmbFreeRAG()
    result = pipeline.query(passage, args.question, top_k=args.top_k)
    print(pipeline.format_response(result))


if __name__ == "__main__":
    main()
