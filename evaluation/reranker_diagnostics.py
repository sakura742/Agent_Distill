"""Diagnostics for verifying that a CrossEncoder reranker is actually active.

This is deliberately separate from the retrieval benchmark so a failed/missing
reranker cannot silently look like a successful embedding-only experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from configs.settings import settings
from knowledge.retriever import get_retriever


def run(
    path: Path,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    rewrite: bool = False,
    embedding_model_name: str | None = None,
    chroma_db_dir: str | None = None,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    details = []
    for row in rows:
        retriever = get_retriever(
            row["domain"],
            embedding_model_name=embedding_model_name,
            chroma_db_dir=chroma_db_dir,
        )
        results = retriever.search(
            row["question"],
            top_k=top_k,
            candidate_k=max(candidate_k, top_k),
            rerank=True,
            hybrid=False,
            rewrite=rewrite,
        )
        scores = [item.rerank_score for item in results]
        details.append({
            "id": row.get("id", row["question"]),
            "question": row["question"],
            "reranker_active": any(score is not None for score in scores),
            "rerank_scores": scores,
            "retrieved_references": [
                f"{item.metadata.get('law_name', '')} {item.metadata.get('article', '')}".strip()
                for item in results
            ],
        })
    return {
        "samples": len(details),
        "reranker_model_configured": bool(settings.reranker_model_name),
        "reranker_model_name": settings.reranker_model_name,
        "embedding_model": embedding_model_name or settings.embedding_model_name,
        "chroma_db_dir": chroma_db_dir or str(settings.chroma_db_dir),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--embedding-model", default=None, help="Embedding 模型名或本地路径")
    parser.add_argument("--chroma-dir", default=None, help="对应 embedding 的独立 Chroma 目录")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run(
        args.benchmark,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rewrite=args.rewrite,
        embedding_model_name=args.embedding_model,
        chroma_db_dir=args.chroma_dir,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
