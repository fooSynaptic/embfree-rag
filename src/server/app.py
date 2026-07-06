"""Flask server for emb-free RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, render_template, request

from src.rag import EmbFreeRAG

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
pipeline = EmbFreeRAG()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/health")
def health():
    return {"status": "ok", "mode": "emb-free-rag"}


@app.route("/answer", methods=["POST"])
def answer():
    payload = request.get_json(force=True, silent=True) or {}
    passage = payload.get("passage", "")
    question = payload.get("question", "")

    if not passage.strip():
        return json.dumps({"answer": "请提供 passage 文本。"}, ensure_ascii=False)

    if not question.strip():
        question = "这段文本的主要话题是什么？"

    result = pipeline.query(passage, question)
    response = {
        "answer": result.answer,
        "context": result.context,
        "topics": {str(k): v for k, v in result.topics.items()},
        "metrics": result.metrics,
        "mode": "emb-free",
    }
    return json.dumps(response, ensure_ascii=False)


def main():
    app.run(debug=True, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
