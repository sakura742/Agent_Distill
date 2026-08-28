"""Paired evaluation -> error analysis -> hard-example mining report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .error_analysis import analyze, write_hard_examples


def relative_reduction(before: float, after: float) -> float | None:
    if before <= 0:
        return None
    return (before - after) / before


def build_iteration_report(raw_result: str, lora_result: str, output: str) -> dict[str, Any]:
    raw = json.loads(Path(raw_result).read_text(encoding="utf-8"))
    lora = json.loads(Path(lora_result).read_text(encoding="utf-8"))
    raw_a = analyze(raw["predictions"])
    lora_a = analyze(lora["predictions"])

    raw_metrics = raw["metrics"]
    lora_metrics = lora["metrics"]
    common = sorted(set(raw_metrics) & set(lora_metrics))
    delta = {m: lora_metrics[m] - raw_metrics[m] for m in common}

    raw_interrupt = raw_metrics.get("interruption_error_rate", raw_metrics.get("error_rate", 0.0))
    lora_interrupt = lora_metrics.get("interruption_error_rate", lora_metrics.get("error_rate", 0.0))
    reduction = relative_reduction(raw_interrupt, lora_interrupt)

    report = {
        "comparison": "Qwen3.5-4B Raw vs Qwen3.5-4B LoRA",
        "raw_metrics": raw_metrics,
        "lora_metrics": lora_metrics,
        "delta_lora_minus_raw": delta,
        "error_reduction": {
            "baseline_error_rate": raw_interrupt,
            "lora_error_rate": lora_interrupt,
            "relative_reduction": reduction,
            "target_35_percent_met": reduction is not None and reduction >= 0.35,
        },
        "raw_errors": raw_a["error_counts"],
        "lora_errors": lora_a["error_counts"],
        "raw_failed_samples": raw_a["failed_samples"],
        "lora_failed_samples": lora_a["failed_samples"],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw")
    p.add_argument("lora")
    p.add_argument("--report", default="data/evaluation/iteration_report.json")
    p.add_argument("--hard-output", default="distill/data/evaluation_hard_examples.jsonl")
    a = p.parse_args()

    build_iteration_report(a.raw, a.lora, a.report)
    lora = json.loads(Path(a.lora).read_text(encoding="utf-8"))
    n = write_hard_examples(lora["predictions"], a.hard_output)
    print(f"hard examples: {n}")


if __name__ == "__main__":
    main()
