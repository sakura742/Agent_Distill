"""Dataset schema and deterministic evaluator for observable Agent behavior."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import (
    accuracy,
    argument_accuracy,
    citation_accuracy,
    citation_precision,
    citation_recall,
    interruption_error,
    mrr,
    recall_at_k,
    token_overlap_f1,
    workflow_success,
)

CATEGORIES = (
    "routing", "retrieval", "tool_calling", "workflow", "answer",
    "answer_citation", "multi_turn",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by = {c: [] for c in CATEGORIES}
    for record in records:
        by.setdefault(record.get("category", "answer"), []).append(record)
    out: dict[str, Any] = {}

    if by["routing"]:
        out["routing_accuracy"] = accuracy(
            [r.get("prediction") for r in by["routing"]],
            [r.get("gold") for r in by["routing"]],
        )
    if by["retrieval"]:
        out["retrieval_recall_at_5"] = recall_at_k(
            [r.get("retrieved", []) for r in by["retrieval"]],
            [r.get("gold", r.get("relevant", [])) for r in by["retrieval"]],
            5,
        )
        out["retrieval_mrr"] = mrr(
            [r.get("retrieved", []) for r in by["retrieval"]],
            [r.get("gold", r.get("relevant", [])) for r in by["retrieval"]],
        )
    if by["tool_calling"]:
        tools = by["tool_calling"]
        out["tool_selection_accuracy"] = accuracy(
            [r.get("prediction", {}).get("name") for r in tools],
            [r.get("gold", {}).get("name") for r in tools],
        )
        out["tool_argument_accuracy"] = sum(
            argument_accuracy(
                r.get("prediction", {}).get("arguments", {}),
                r.get("gold", {}).get("arguments", {}),
            ) for r in tools
        ) / len(tools)
    if by["workflow"]:
        out["workflow_success_rate"] = sum(
            workflow_success(r.get("trace", []), r.get("verification", {}))
            for r in by["workflow"]
        ) / len(by["workflow"])
        out["interruption_error_rate"] = sum(
            interruption_error(r.get("trace", []), r.get("verification", {}))
            for r in by["workflow"]
        ) / len(by["workflow"])
    if by["answer"]:
        out["answer_token_overlap_f1"] = sum(
            token_overlap_f1(r.get("prediction", ""), r.get("gold", ""))
            for r in by["answer"]
        ) / len(by["answer"])
    if by["answer_citation"]:
        citations = by["answer_citation"]
        out["citation_precision"] = sum(
            citation_precision(r.get("predicted", r.get("citations", [])), r.get("gold", []))
            for r in citations
        ) / len(citations)
        out["citation_recall"] = sum(
            citation_recall(r.get("predicted", r.get("citations", [])), r.get("gold", []))
            for r in citations
        ) / len(citations)
        out["citation_accuracy"] = sum(
            citation_accuracy(r.get("predicted", r.get("citations", [])), r.get("gold", []))
            for r in citations
        ) / len(citations)
    if by["multi_turn"]:
        out["multi_turn_success_rate"] = accuracy(
            [r.get("prediction") for r in by["multi_turn"]],
            [r.get("gold") for r in by["multi_turn"]],
        )
    return out
