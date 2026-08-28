"""Paired evaluation -> error analysis -> hard-example mining report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .error_analysis import analyze


def relative_reduction(before: float, after: float) -> float | None:
    if before <= 0:
        return None
    return (before - after) / before


def _paired(raw_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_id = {x.get("id"): x for x in raw_rows}
    lora_by_id = {x.get("id"): x for x in lora_rows}
    result = []
    for case_id in sorted(set(raw_by_id) | set(lora_by_id)):
        r, l = raw_by_id.get(case_id), lora_by_id.get(case_id)
        if r is None or l is None:
            continue
        result.append({
            "id": case_id,
            "category": r.get("category", l.get("category")),
            "raw_error": bool(r.get("error")),
            "lora_error": bool(l.get("error")),
            "raw_retry_count": r.get("retry_count", 0),
            "lora_retry_count": l.get("retry_count", 0),
            "raw_prediction": r.get("prediction"),
            "lora_prediction": l.get("prediction"),
        })
    return result


def _hard_examples(raw_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_id = {x.get("id"): x for x in raw_rows}
    lora_by_id = {x.get("id"): x for x in lora_rows}
    items = []
    for case_id in sorted(set(raw_by_id) | set(lora_by_id)):
        r, l = raw_by_id.get(case_id), lora_by_id.get(case_id)
        if not r or not l:
            continue
        re = analyze([r])["failed_samples"] > 0
        le = analyze([l])["failed_samples"] > 0
        if not (re or le):
            continue
        item = dict(l)
        item["hard_example_reason"] = (
            "raw_failed_lora_fixed" if re and not le else
            "both_failed" if re and le else "lora_regression"
        )
        item["raw_error_types"] = analyze([r])["failed_samples"] and analyze([r])["hard_examples"][0].get("error_types", [])
        item["lora_error_types"] = analyze([l])["failed_samples"] and analyze([l])["hard_examples"][0].get("error_types", [])
        items.append(item)
    return items


def build_iteration_report(raw_result: str, lora_result: str, output: str, hard_output: str | None = None) -> dict[str, Any]:
    raw = json.loads(Path(raw_result).read_text(encoding="utf-8"))
    lora = json.loads(Path(lora_result).read_text(encoding="utf-8"))
    raw_a, lora_a = analyze(raw["predictions"]), analyze(lora["predictions"])

    raw_metrics, lora_metrics = raw["metrics"], lora["metrics"]
    common = sorted(set(raw_metrics) & set(lora_metrics))
    delta = {m: lora_metrics[m] - raw_metrics[m] for m in common}
    raw_interrupt = raw_metrics.get("interruption_error_rate", raw_metrics.get("error_rate", 0.0))
    lora_interrupt = lora_metrics.get("interruption_error_rate", lora_metrics.get("error_rate", 0.0))
    reduction = relative_reduction(raw_interrupt, lora_interrupt)

    report = {
        "comparison": "Qwen3.5-4B Raw vs Qwen3.5-4B LoRA",
        "benchmark_samples": len(raw["predictions"]),
        "paired_samples": len(_paired(raw["predictions"], lora["predictions"])),
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
        "paired_cases": _paired(raw["predictions"], lora["predictions"]),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if hard_output:
        hp = Path(hard_output)
        hp.parent.mkdir(parents=True, exist_ok=True)
        with hp.open("w", encoding="utf-8") as f:
            for row in _hard_examples(raw["predictions"], lora["predictions"]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["hard_examples"] = len(_hard_examples(raw["predictions"], lora["predictions"]))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw")
    p.add_argument("lora")
    p.add_argument("--report", default="data/evaluation/iteration_report.json")
    p.add_argument("--hard-output", default="distill/data/evaluation_hard_examples.jsonl")
    a = p.parse_args()
    report = build_iteration_report(a.raw, a.lora, a.report, a.hard_output)
    print(json.dumps(report["error_reduction"], ensure_ascii=False, indent=2))
    print(f"hard examples: {report.get('hard_examples', 0)}")


if __name__ == "__main__":
    main()
