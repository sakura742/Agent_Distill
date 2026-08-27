"""Phase 5: collect teacher trajectories from the Agent Runtime.

A trajectory is a structured supervision record, not a raw chain-of-thought dump.
It captures the observable decisions needed by a small Agent: route, plan, tool
call, retrieved evidence, verification result, and final answer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.runtime.graph import build_legal_agent_graph
from configs.settings import settings


TEACHER_SYSTEM = (
    "你是法律 Agent 的教师模型。请根据用户问题和检索到的法律依据，"
    "生成准确、克制、可执行的法律建议。不要编造法条；证据不足时明确说明。"
)


def _teacher_generator(client: OpenAI):
    def generate(question: str, evidence: str, citations: list[dict[str, Any]]) -> str:
        prompt = (
            f"用户问题：{question}\n\n"
            f"检索证据：\n{evidence}\n\n"
            "请只返回最终法律建议，不输出隐藏思维过程。"
        )
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=settings.trajectory_max_tokens,
        )
        return response.choices[0].message.content.strip()

    return generate


def collect_trajectory(question: str, client: OpenAI | None = None) -> dict[str, Any]:
    client = client or OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    graph = build_legal_agent_graph(answer_generator=_teacher_generator(client))
    state = graph.invoke({"question": question, "retry_count": 0, "trace": []})
    return {
        "question": question,
        "domain": state.get("domain"),
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence", 0.0),
        "plan": state.get("plan", []),
        "tool": {
            "name": state.get("tool_name"),
            "arguments": state.get("tool_arguments", {}),
        },
        "retrieved_documents": state.get("retrieved_documents", []),
        "citations": state.get("citations", []),
        "answer": state.get("answer", ""),
        "verification": state.get("verification", {}),
        "retry_count": state.get("retry_count", 0),
        "trace": state.get("trace", []),
    }


def write_trajectories(questions: list[str], output: Path | None = None) -> int:
    if not settings.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    output = output or settings.trajectory_data_path
    output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    count = 0
    with output.open("a", encoding="utf-8") as f:
        for question in questions:
            record = collect_trajectory(question, client)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_questions(path: str | os.PathLike[str]) -> list[str]:
    questions: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions.append(item.get("input") or item.get("question") or item.get("user_query"))
    return [q for q in questions if q]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="JSONL with input/question/user_query field")
    args = parser.parse_args()
    print(f"写入 {write_trajectories(load_questions(args.questions))} 条 trajectory")
