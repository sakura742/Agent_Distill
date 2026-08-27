"""Phase 5 hard-example mining from trajectory/evaluation signals."""

from __future__ import annotations

import json
from pathlib import Path

from configs.settings import settings


def is_hard(record: dict) -> bool:
    verification = record.get("verification", {})
    if verification and not verification.get("passed", False):
        return True
    if record.get("retry_count", 0) > 0:
        return True
    if float(record.get("intent_confidence", 1.0)) < 0.55:
        return True
    tool = record.get("tool", {})
    if not tool.get("name") or not tool.get("arguments", {}).get("query"):
        return True
    if not record.get("citations"):
        return True
    return False


def mine(input_path: Path | None = None, output_path: Path | None = None) -> int:
    input_path = input_path or settings.trajectory_data_path
    output_path = output_path or settings.hard_example_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if is_hard(record):
                record["hard_reason"] = _reason(record)
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def _reason(record: dict) -> str:
    verification = record.get("verification", {})
    if verification and not verification.get("passed", False):
        return verification.get("reason", "verification_failed")
    if record.get("retry_count", 0) > 0:
        return "runtime_retry"
    if float(record.get("intent_confidence", 1.0)) < 0.55:
        return "low_route_confidence"
    if not record.get("citations"):
        return "missing_citations"
    return "invalid_tool_call"


if __name__ == "__main__":
    print(f"挖掘困难样本：{mine()} 条")
