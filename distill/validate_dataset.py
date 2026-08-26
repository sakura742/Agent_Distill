"""Validate and normalize structured Agent trajectory JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from distill.trajectory_schema import validate_trajectory


def validate_file(path: Path) -> tuple[int, int, list[str]]:
    total = valid = 0
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            row_errors = validate_trajectory(payload)
            if row_errors:
                errors.extend(f"line {line_no}: {item}" for item in row_errors)
            else:
                valid += 1
    return total, valid, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    total, valid, errors = validate_file(args.path)
    print(f"total={total} valid={valid} invalid={total - valid}")
    for error in errors[:50]:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
