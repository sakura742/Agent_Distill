"""Collect teacher trajectories against the real legal tool runtime.

Unlike synthetic tool-call generation, this module executes every teacher-selected
call through the same ToolExecutor abstraction used by the Agent Runtime. The
resulting observations are therefore real and replayable. No private chain-of-
thought is stored: only concise actions, tool calls, observations and the final
answer are persisted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from configs.settings import settings
from .trajectory_schema import AgentTrajectory, AgentStep, ToolCall, validate_trajectory

_SYSTEM = """你是法律 Agent 教师模型。请完成用户的法律咨询任务。
你可以调用提供的法律检索工具；只有在需要法律依据时调用工具。
每次调用工具后，根据真实工具结果决定下一步，最多执行 {max_steps} 次工具调用。
最终给出简洁、基于检索证据的回答。
不要输出或记录长篇思维链，只保留可执行的 action 摘要。
"""


def load_tool_contract(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tools = payload.get("tools", payload)
    if not isinstance(tools, list):
        raise ValueError("tool contract must contain a list of tools")
    return tools


def _openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return result


def collect_trajectory(
    user_query: str,
    tools: list[dict[str, Any]],
    execute: Callable[[str, dict[str, Any]], str],
    *,
    max_steps: int = 4,
) -> AgentTrajectory:
    """Run a teacher/tool interaction loop and return a canonical trajectory."""
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for teacher generation")
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM.format(max_steps=max_steps)},
        {"role": "user", "content": user_query},
    ]
    steps: list[AgentStep] = []
    intent = "legal_consultation"
    domain = None

    for step_no in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            tools=_openai_tools(tools),
            tool_choice="auto",
            temperature=0.1,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            final = message.content or ""
            payload = AgentTrajectory(
                user_query=user_query,
                intent=intent,
                domain=domain,
                steps=steps,
                final_answer=final,
            )
            errors = validate_trajectory(payload.to_dict())
            if errors:
                raise ValueError("invalid collected trajectory: " + "; ".join(errors))
            return payload

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [call.model_dump() for call in tool_calls],
        })
        for call in tool_calls:
            name = call.function.name
            arguments = json.loads(call.function.arguments or "{}")
            observation = execute(name, arguments)
            if name.startswith("search_labor"):
                domain = "labor"
            elif name.startswith("search_civil"):
                domain = "civil"
            steps.append(AgentStep(
                step=step_no,
                action=f"调用法律检索工具获取与问题相关的依据",
                tool_call=ToolCall(name=name, arguments=arguments),
                observation=observation,
            ))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": observation,
            })

    raise RuntimeError(f"teacher did not finish within max_steps={max_steps}")


def collect_to_jsonl(
    queries: list[str],
    tools: list[dict[str, Any]],
    execute: Callable[[str, dict[str, Any]], str],
    output: Path,
    *,
    max_steps: int = 4,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for query in queries:
            trajectory = collect_trajectory(query, tools, execute, max_steps=max_steps)
            handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
