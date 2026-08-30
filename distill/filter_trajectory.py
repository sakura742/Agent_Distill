"""Filter trajectory JSONL into accepted and rejected datasets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def filter_trajectories(source: Path, accepted: Path, rejected: Path) -> tuple[int, int]:
    good = bad = 0
    accepted.parent.mkdir(parents=True, exist_ok=True)
    rejected.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as src, accepted.open("w", encoding="utf-8") as good_file, rejected.open("w", encoding="utf-8") as bad_file:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            answer_ok = bool(str(row.get("answer", "")).strip())
            verification = row.get("verification") or {}
            verification_ok = verification.get("passed") is True
            decision_ok = row.get("domain") not in (None, "unknown") and bool((row.get("tool") or {}).get("name"))
            target = good_file if answer_ok and verification_ok and decision_ok else bad_file
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            if target is good_file:
                good += 1
            else:
                bad += 1
    return good, bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    args = parser.parse_args()
    good, bad = filter_trajectories(args.source, args.accepted, args.rejected)
    print(f"accepted={good} rejected={bad}")


if __name__ == "__main__":
    main()
