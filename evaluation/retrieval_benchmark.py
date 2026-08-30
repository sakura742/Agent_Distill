"""Gold-based retrieval benchmark for Phase 6.

The benchmark deliberately expects canonical references such as
``中华人民共和国劳动法及相关法律 四十四`` rather than historical placeholder
IDs (for example ``labor_wage_regulation``).  Run it only after the local
Chroma collections have been built.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from knowledge.retriever import get_retriever


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("question") or not row.get("domain") or not row.get("gold_references"):
                raise ValueError(f"{path}:{line_no} 缺少 question/domain/gold_references")
            rows.append(row)
    return rows


def _canonical_reference(metadata: dict[str, Any]) -> str:
    law = metadata.get("law_name", "")
    article = metadata.get("article", "")
    return f"{law} {article}".strip()


def evaluate(rows: list[dict[str, Any]], top_k: int = 5, rerank: bool = False) -> dict[str, Any]:
    per_row = []
    for row in rows:
        retriever = get_retriever(row["domain"])
        results = retriever.search(
            row["question"],
            top_k=top_k,
            candidate_k=max(top_k * 4, top_k),
            rerank=rerank,
        )
        retrieved = [_canonical_reference(item.metadata) for item in results]
        retrieved_set = set(retrieved)
        gold = set(row["gold_references"])
        hits = retrieved_set & gold
        first_rank = next((i for i, ref in enumerate(retrieved, 1) if ref in gold), None)
        per_row.append({
            "id": row.get("id", row["question"]),
            "question": row["question"],
            "domain": row["domain"],
            "gold_references": sorted(gold),
            "retrieved_references": retrieved,
            "hit_count": len(hits),
            "precision_at_k": len(hits) / len(retrieved) if retrieved else 0.0,
            "recall_at_k": len(hits) / len(gold) if gold else 0.0,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "scores": [item.score for item in results],
            "distances": [item.distance for item in results],
            "rerank_scores": [item.rerank_score for item in results],
        })
    n = len(per_row)
    return {
        "samples": n,
        "top_k": top_k,
        "rerank": rerank,
        "precision_at_k": sum(x["precision_at_k"] for x in per_row) / n if n else 0.0,
        "recall_at_k": sum(x["recall_at_k"] for x in per_row) / n if n else 0.0,
        "mrr": sum(x["reciprocal_rank"] for x in per_row) / n if n else 0.0,
        "details": per_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(_load(args.benchmark), top_k=args.top_k, rerank=args.rerank)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
