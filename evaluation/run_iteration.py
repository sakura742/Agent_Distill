"""Run the Phase 6 evaluation/analysis loop on existing Raw/LoRA results."""
from __future__ import annotations

import argparse
from pathlib import Path

from .iteration import build_iteration_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired Raw vs LoRA evaluation report")
    parser.add_argument("--raw", default="data/evaluation/results/qwen35_4b_raw.json")
    parser.add_argument("--lora", default="data/evaluation/results/qwen35_4b_lora.json")
    parser.add_argument("--report", default="data/evaluation/iteration_report.json")
    args = parser.parse_args()

    for path in (args.raw, args.lora):
        if not Path(path).exists():
            raise FileNotFoundError(f"Benchmark result not found: {path}. Run evaluation.run_qwen35_benchmark first.")

    report = build_iteration_report(args.raw, args.lora, args.report)
    reduction = report["error_reduction"]["relative_reduction"]
    if reduction is None:
        print("35% target: not measurable (baseline interruption/error rate is zero)")
    else:
        print(f"35% target: {'MET' if reduction >= 0.35 else 'NOT MET'} ({reduction:.2%})")


if __name__ == "__main__":
    main()
