"""Phase 6 error taxonomy and hard-example export.

Consumes observable per-sample benchmark predictions. Error labels are
category-specific so the exported hard examples can be fed back into Phase 5.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def classify(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    category = row.get("category")
    gold = row.get("gold")

    if category == "routing" and row.get("route_prediction") != gold:
        errors.append("routing_error")

    if category == "tool_calling":
        g = gold or {}
        p = row.get("tool_prediction") or {}
        if p.get("name") != g.get("name"):
            errors.append("tool_selection_error")
        if p.get("arguments") != g.get("arguments"):
            errors.append("tool_parameter_error")

    if category == "retrieval":
        relevant = set(gold or row.get("relevant", []))
        retrieved = set((row.get("retrieved") or [])[:5])
        if relevant and not relevant & retrieved:
            errors.append("retrieval_miss")

    if category == "workflow":
        required = {
            "intent_analysis", "task_planning", "tool_decision",
            "tool_execution", "retrieval", "generation", "verification",
        }
        nodes = {x.get("node") for x in row.get("trace", [])}
        if not required.issubset(nodes) or not row.get("verification", {}).get("passed", False):
            errors.append("workflow_error")

    if category == "answer" and not str(row.get("prediction") or "").strip():
        errors.append("empty_answer")

    if category == "answer_citation":
        predicted = set(row.get("citations") or row.get("predicted") or [])
        expected = set(gold or [])
        if not predicted:
            errors.append("missing_citation")
        elif not predicted & expected:
            errors.append("wrong_citation")
        elif predicted != expected:
            errors.append("citation_partial_match")

    if row.get("error"):
        errors.append("runtime_error")
    if row.get("verification") and not row["verification"].get("passed", False):
        errors.append("verification_failure")
    if row.get("retry_count", 0) > 0:
        errors.append("retry_or_interruption")
    return errors


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    hard: list[dict[str, Any]] = []
    for row in records:
        errors = classify(row)
        counts.update(errors)
        if errors:
            item = dict(row)
            item["error_types"] = errors
            item["hard_score"] = len(errors)
            hard.append(item)
    hard.sort(key=lambda x: (-x["hard_score"], x.get("id", "")))
    return {
        "samples": len(records),
        "failed_samples": len(hard),
        "error_counts": dict(counts),
        "hard_examples": hard,
    }


def write_hard_examples(records: list[dict[str, Any]], output: str | Path) -> int:
    result = analyze(records)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in result["hard_examples"]:
            # Preserve the full observable failure for audit/retraining tooling.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(result["hard_examples"])


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--hard-output", default="distill/data/evaluation_hard_examples.jsonl")
    p.add_argument("--report", default="data/evaluation/error_analysis.json")
    a = p.parse_args()
    rows = json.loads(Path(a.results).read_text(encoding="utf-8"))["predictions"]
    result = analyze(rows)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_hard_examples(rows, a.hard_output)
    print(json.dumps({
        "samples": result["samples"],
        "failed_samples": result["failed_samples"],
        "error_counts": result["error_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
