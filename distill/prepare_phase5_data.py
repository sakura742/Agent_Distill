"""Prepare validated Phase 5 decision/answer SFT datasets."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from configs.settings import settings


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"trajectory 文件不存在: {path}")
    rows = []
    seen = set()
    rejected = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            q = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not q or q in seen:
                continue
            seen.add(q)
            verification = row.get("verification") or {}
            if not answer or verification.get("passed") is not True:
                rejected += 1
                continue
            domain = row.get("domain")
            tool_name = (row.get("tool") or {}).get("name")
            valid_decision = bool(tool_name) if domain not in (None, "unknown") else not tool_name
            if not valid_decision:
                rejected += 1
                continue
            rows.append(row)
    if not rows:
        raise ValueError("没有可用于 SFT 的有效 trajectory")
    print(f"有效 trajectory: {len(rows)}；过滤失败/空答案/非法路由: {rejected}")
    return rows


def _decision(row):
    tool = row.get("tool") or {}
    target = {
        "domain": row.get("domain"),
        "intent": row.get("intent"),
        "tool": {
            "name": tool.get("name"),
            "arguments": tool.get("arguments", {}),
        } if tool.get("name") else None,
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": "你是法律 Agent。根据用户问题选择法域、意图和合适的法律检索工具。非法律问题或无需法律检索时，选择 unknown 并将 tool 设为 null。只输出合法 JSON，不要解释。",
            },
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def _answer(row):
    evidence = row.get("tool_result", "")
    if not evidence:
        evidence = "\n\n".join(
            f"[{d.get('reference', '')}]\n{d.get('content', '')}"
            for d in row.get("retrieved_documents", [])
        )
    return {
        "messages": [
            {
                "role": "system",
                "content": "你是法律 Agent。法律问题只能依据提供的检索证据回答；证据不足时明确说明。不要编造法条或事实。非法律问题直接自然回复，不调用法律工具，不强行提供法律分析。",
            },
            {
                "role": "user",
                "content": f"问题：{row['question']}\n\n法律证据：\n{evidence}",
            },
            {"role": "assistant", "content": str(row.get("answer", "")).strip()},
        ]
    }


def prepare():
    rows = _load(settings.trajectory_data_path)
    dp, ap = settings.phase5_decision_data_path, settings.phase5_answer_data_path
    dp.parent.mkdir(parents=True, exist_ok=True)
    ap.parent.mkdir(parents=True, exist_ok=True)
    decisions = [_decision(r) for r in rows]
    answers = [_answer(r) for r in rows]
    for path, data in ((dp, decisions), (ap, answers)):
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Decision SFT: {len(decisions)} -> {dp}")
    print(f"Answer SFT: {len(answers)} -> {ap}")
    return len(decisions), len(answers)


if __name__ == "__main__":
    prepare()
