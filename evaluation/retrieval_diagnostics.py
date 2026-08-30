"""Retrieve a large candidate pool and report whether gold is reachable before reranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from knowledge.retriever import get_retriever


def run(path: Path, *, candidate_k: int = 50, rewrite: bool = False, embedding_model: str | None = None, chroma_dir: str | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    details = []
    for row in rows:
        retriever = get_retriever(row["domain"], embedding_model_name=embedding_model, chroma_db_dir=chroma_dir)
        results = retriever.search(row["question"], top_k=candidate_k, candidate_k=candidate_k, rewrite=rewrite, rerank=False, hybrid=False)
        refs = [f"{x.metadata.get('law_name','')} {x.metadata.get('article','')}".strip() for x in results]
        gold = set(row["gold_references"])
        ranks = {ref: (refs.index(ref) + 1) for ref in gold if ref in refs}
        details.append({
            "id": row.get("id", row["question"]),
            "question": row["question"],
            "gold_references": sorted(gold),
            "gold_in_candidate_pool": bool(ranks),
            "gold_ranks": ranks,
            "retrieved_references": refs,
            "scores": [x.score for x in results],
        })
    return {"samples": len(details), "candidate_k": candidate_k, "rewrite": rewrite, "details": details}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", type=Path)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--chroma-dir", default=None)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    report = run(a.benchmark, candidate_k=a.candidate_k, rewrite=a.rewrite, embedding_model=a.embedding_model, chroma_dir=a.chroma_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if a.output:
        a.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
