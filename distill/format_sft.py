"""Convert validated AgentTrajectory JSONL into supervised training records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def trajectory_to_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "你是法律领域 Agent。先判断意图和法域，再选择合适工具完成任务。"},
        {"role": "user", "content": item["user_query"]},
    ]

    for step in item.get("steps", []):
        action = step.get("action")
        if action:
            messages.append({"role": "assistant", "content": action})
        call = step.get("tool_call")
        if call:
            messages.append({
                "role": "assistant",
                "content": json.dumps(
                    {"tool": call["name"], "arguments": call.get("arguments", {})},
                    ensure_ascii=False,
                ),
            })
        observation = step.get("observation")
        if observation:
            messages.append({"role": "tool", "content": observation})

    messages.append({"role": "assistant", "content": item["final_answer"]})
    return messages


def convert_jsonl(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            messages = trajectory_to_messages(item)
            dst.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    return count
