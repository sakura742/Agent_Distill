"""Phase 5: collect teacher trajectories from the Agent Runtime.

A trajectory is structured supervision data. It records observable routing,
tool use, evidence, selected citations, verification, and final answer.
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
    "对于法律问题，必须在回答中明确写出实际使用的法条编号，例如“依据《劳动法》第四十四条”。"
    "只引用真正支撑结论的检索证据，不要为了凑引用数量而引用无关法条。"
    "如果用户只是寒暄、闲聊或非法律问题，不要进行法律检索，也不要引用任何法条。"
)


def _teacher_generator(client: OpenAI):
    def generate(question: str, evidence: str, citations: list[dict[str, Any]]) -> str:
        prompt = (
            f"用户问题：{question}\n\n"
            f"检索证据（可能包含噪声，必须自行筛选）：\n{evidence}\n\n"
            "请只返回最终法律建议，不输出隐藏思维过程。法律问题请明确引用实际使用的法条编号；"
            "不要把所有检索结果都当作引用。非法律问题直接自然回复。"
        )
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=settings.trajectory_max_tokens,
        )
        return response.choices[0].message.content.strip()
    return generate


def collect_trajectory(question: str, client: OpenAI | None = None) -> dict[str, Any]:
    client = client or OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    graph = build_legal_agent_graph(answer_generator=_teacher_generator(client))
    state = graph.invoke({"question": question, "retry_count": 0, "trace": []})
    return {
        "question": question,
        "domain": state.get("domain"),
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence", 0.0),
        "plan": state.get("plan", []),
        "tool": {"name": state.get("tool_name"), "arguments": state.get("tool_arguments", {})},
        "retrieved_documents": state.get("retrieved_documents", []),
        "citations": state.get("citations", []),
        "answer": state.get("answer", ""),
        "verification": state.get("verification", {}),
        "retry_count": state.get("retry_count", 0),
        "trace": state.get("trace", []),
    }


def write_trajectories(questions: list[str], output: Path | None = None, append: bool = False) -> int:
    if not settings.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    if not questions:
        raise ValueError("输入问题为空：请检查 JSONL 字段是否为 input/question/user_query")
    output = output or settings.trajectory_data_path
    output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    count = 0
    mode = "a" if append else "w"
    with output.open(mode, encoding="utf-8") as f:
        for index, question in enumerate(questions, 1):
            if not question or not question.strip():
                continue
            try:
                record = collect_trajectory(question, client)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                print(f"[{index}/{len(questions)}] trajectory generated")
            except Exception as exc:
                raise RuntimeError(f"第 {index} 条 trajectory 生成失败：{question[:80]}") from exc
    return count


def load_questions(path: str | os.PathLike[str]) -> list[str]:
    questions: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("input") or item.get("question") or item.get("user_query")
            if question:
                questions.append(str(question))
            else:
                raise ValueError(f"第 {line_no} 行缺少 input/question/user_query 字段")
    return questions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="JSONL with input/question/user_query field")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--append", action="store_true", help="追加到已有 trajectory 文件；默认覆盖")
    args = parser.parse_args()
    questions = load_questions(args.questions)
    print(f"读取 {len(questions)} 条问题")
    print(f"写入 {write_trajectories(questions, args.output, args.append)} 条 trajectory")
