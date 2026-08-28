"""Validate Phase 6 benchmark structure before running expensive model inference."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = {"id", "category", "input", "expected"}
EXPECTED_CATEGORIES = {"routing", "retrieval", "tool_calling", "workflow", "answer", "citation"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/evaluation/benchmark.jsonl")
    args = parser.parse_args()
    path = Path(args.benchmark)
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            missing = REQUIRED - obj.keys()
            if missing:
                raise ValueError(f"line {n}: missing {sorted(missing)}")
            rows.append(obj)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark ids must be unique")
    counts = Counter(r["category"] for r in rows)
    unknown = set(counts) - EXPECTED_CATEGORIES
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    print(f"valid: {len(rows)} cases")
    print("categories:", dict(sorted(counts.items())))
    print("status: PASS")


if __name__ == "__main__":
    main()
