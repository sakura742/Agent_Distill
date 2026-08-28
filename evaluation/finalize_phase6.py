"""Validate and finalize a Phase 6 Raw-vs-LoRA experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_METRICS = {
    "routing_accuracy", "retrieval_recall_at_5", "retrieval_mrr",
    "tool_selection_accuracy", "tool_argument_accuracy", "workflow_success_rate",
    "interruption_error_rate", "answer_token_overlap_f1", "citation_precision",
    "citation_recall", "citation_accuracy", "avg_latency_ms", "error_rate",
}


def validate_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_METRICS - set(data.get("metrics", {})))
    predictions = data.get("predictions", [])
    ids = [x.get("id") for x in predictions]
    if missing:
        raise ValueError(f"{path}: missing metrics: {missing}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate prediction ids")
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/evaluation/results/qwen35_4b_raw.json")
    p.add_argument("--lora", default="data/evaluation/results/qwen35_4b_lora.json")
    p.add_argument("--report", default="data/evaluation/iteration_report.json")
    a = p.parse_args()

    raw, lora = validate_result(Path(a.raw)), validate_result(Path(a.lora))
    raw_ids, lora_ids = {x["id"] for x in raw["predictions"]}, {x["id"] for x in lora["predictions"]}
    if raw_ids != lora_ids:
        raise ValueError("Raw and LoRA results do not contain the same benchmark IDs")

    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    reduction = report["error_reduction"]["relative_reduction"]
    print(f"paired cases: {len(raw_ids)}")
    print(f"error reduction: {'N/A' if reduction is None else f'{reduction:.2%}'}")
    print(f"35% target: {'MET' if report['error_reduction']['target_35_percent_met'] else 'NOT MET'}")
    print("Phase 6 empirical result validation: PASS")


if __name__ == "__main__":
    main()
