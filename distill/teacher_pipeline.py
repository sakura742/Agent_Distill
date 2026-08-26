"""Teacher trajectory generation pipeline.

DeepSeek is used as the teacher for planning/tool-choice supervision. The output is a
canonical trajectory JSONL file, deliberately separated from the SFT conversion step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from configs.settings import settings

SYSTEM_PROMPT = """你是法律 Agent 教师模型。针对用户问题，输出严格 JSON 对象：
{
  \"user_query\": string,
  \"intent\": string,
  \"domain\": \"civil\" | \"labor\" | null,
  \"steps\": [
    {\"step\": 1, \"action\": string, \"tool_call\": {\"name\": string, \"arguments\": object} | null, \"observation\": null}
  ],
  \"final_answer\": string | null
}
只描述可执行的 Agent 决策，不输出长篇思维链；把关键决策写在 action 字段中。
工具名称必须来自提供的工具契约。"""


def build_prompt(user_query: str, tools: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n工具契约：" + json.dumps(tools, ensure_ascii=False)},
        {"role": "user", "content": user_query},
    ]


def generate_trajectory(user_query: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for teacher generation")
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=build_prompt(user_query, tools),
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def generate_file(queries: list[str], tools: list[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for query in queries:
            trajectory = generate_trajectory(query, tools)
            handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            count += 1
    return count
