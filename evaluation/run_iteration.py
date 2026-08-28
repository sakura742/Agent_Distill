"""Run the Phase 6 paired evaluation -> error analysis -> hard-example loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .iteration import build_iteration_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired Raw vs LoRA evaluation report")
    parser.add_argument("--raw", default="data/evaluation/results/qwen35_4b_raw.json")
    parser.add_argument("--lora", default="data/evaluation/results/qwen35_4b_lora.json")
    parser.add_argument("--report", default="data/evaluation/iteration_report.json")
    parser.add_argument("--hard-output", default="distill/data/evaluation_hard_examples.jsonl")
    args = parser.parse_args()

    for path in (args.raw, args.lora):
        if not Path(path).exists():
            raise FileNotFoundError(f"Benchmark result not found: {path}. Run evaluation.run_qwen35_benchmark first.")

    report = build_iteration_report(args.raw, args.lora, args.report, args.hard_output)
    print(json.dumps({
        "paired_samples": report["paired_samples"],
        "hard_examples": report.get("hard_examples", 0),
        "error_reduction": report["error_reduction"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
