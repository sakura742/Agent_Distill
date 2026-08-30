# -*- coding: utf-8 -*-
"""RAG 检索评估指标：Recall@K 与 MRR。"""

from __future__ import annotations

from typing import Iterable


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: set[str], k: int) -> float:
    """Return the fraction of relevant documents found in the top-k results."""
    if not relevant_ids or k <= 0:
        return 0.0
    retrieved = set(list(retrieved_ids)[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: Iterable[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(records: list[dict], k: int = 5) -> dict[str, float]:
    if not records:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    recalls = []
    reciprocal_ranks = []
    for record in records:
        relevant = set(record.get("relevant_ids", []))
        retrieved = record.get("retrieved_ids", [])
        recalls.append(recall_at_k(retrieved, relevant, k))
        reciprocal_ranks.append(mrr(retrieved, relevant))
    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }
