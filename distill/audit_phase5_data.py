"""Audit Phase 5 SFT JSONL before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit(path: Path) -> dict[str, Any]:
    rows = _load(path)
    questions = [str(r.get("question") or "").strip() for r in rows]
    answers = [str(r.get("answer") or "").strip() for r in rows]
    domains = {}
    duplicate_count = len(questions) - len(set(q for q in questions if q))
    for row in rows:
        domain = row.get("domain") or "missing"
        domains[domain] = domains.get(domain, 0) + 1
    return {
        "path": str(path),
        "rows": len(rows),
        "empty_question": sum(not q for q in questions),
        "empty_answer": sum(not a for a in answers),
        "duplicate_questions": duplicate_count,
        "domains": domains,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        report = audit(path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["rows"] == 0:
            raise SystemExit(f"数据为空: {path}")
        if report["empty_question"] or report["empty_answer"] or report["duplicate_questions"]:
            raise SystemExit(f"数据审计失败: {path}")


if __name__ == "__main__":
    main()
