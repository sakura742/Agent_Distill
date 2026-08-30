"""Audit Phase 5 SFT JSONL before training.

Supports both validated trajectory rows and the final chat-message SFT format
produced by ``prepare_phase5_data.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _message_value(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content") or "").strip()
    return ""


def _question_answer(row: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return question, answer, and inferred domain for trajectory or chat rows."""
    if isinstance(row.get("messages"), list):
        question = _message_value(row, "user")
        answer = _message_value(row, "assistant")
        return question, answer, None
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    return question, answer, row.get("domain")


def audit(path: Path) -> dict[str, Any]:
    rows = _load(path)
    questions: list[str] = []
    answers: list[str] = []
    domains: dict[str, int] = {}
    for row in rows:
        question, answer, domain = _question_answer(row)
        questions.append(question)
        answers.append(answer)
        if domain is not None:
            key = str(domain or "missing")
            domains[key] = domains.get(key, 0) + 1

    non_empty_questions = [q for q in questions if q]
    return {
        "path": str(path),
        "rows": len(rows),
        "empty_question": sum(not q for q in questions),
        "empty_answer": sum(not a for a in answers),
        "duplicate_questions": len(non_empty_questions) - len(set(non_empty_questions)),
        "domains": domains,
        "format": "chat_messages" if any(isinstance(r.get("messages"), list) for r in rows) else "trajectory",
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
