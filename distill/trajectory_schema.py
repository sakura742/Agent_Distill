"""Canonical Agent trajectory schema used by teacher generation, training and evaluation.

The schema intentionally stores actions and observations separately so the same trace can be
replayed by LangGraph/MCP and converted into an SFT example without losing structure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    step: int
    action: str
    tool_call: Optional[ToolCall] = None
    observation: Optional[Any] = None


@dataclass
class AgentTrajectory:
    user_query: str
    intent: str
    domain: Optional[str]
    steps: list[AgentStep]
    final_answer: Optional[str] = None
    source: str = "teacher"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_sft_text(self) -> str:
        """Convert the structured trace into the compact training representation."""
        lines = [f"用户问题：{self.user_query}", f"意图：{self.intent}"]
        if self.domain:
            lines.append(f"法域：{self.domain}")
        for step in self.steps:
            lines.append(f"步骤 {step.step}：{step.action}")
            if step.tool_call:
                lines.append(
                    "工具调用：" + step.tool_call.name + " " + _json(step.tool_call.arguments)
                )
            if step.observation is not None:
                lines.append("工具结果：" + _json(step.observation))
        if self.final_answer:
            lines.append(f"最终回答：{self.final_answer}")
        return "\n".join(lines)


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_trajectory(payload: dict[str, Any]) -> list[str]:
    """Return validation errors instead of raising, making dataset QA batch-friendly."""
    errors: list[str] = []
    for key in ("user_query", "intent", "steps"):
        if key not in payload:
            errors.append(f"missing field: {key}")
    if not isinstance(payload.get("user_query"), str) or not payload.get("user_query", "").strip():
        errors.append("user_query must be a non-empty string")
    if not isinstance(payload.get("steps"), list) or not payload.get("steps"):
        errors.append("steps must be a non-empty list")
    for index, step in enumerate(payload.get("steps", [])):
        if not isinstance(step, dict):
            errors.append(f"steps[{index}] must be an object")
            continue
        if not isinstance(step.get("step"), int):
            errors.append(f"steps[{index}].step must be an integer")
        if not isinstance(step.get("action"), str) or not step.get("action", "").strip():
            errors.append(f"steps[{index}].action must be a non-empty string")
        call = step.get("tool_call")
        if call is not None:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                errors.append(f"steps[{index}].tool_call must contain name")
            elif not isinstance(call.get("arguments", {}), dict):
                errors.append(f"steps[{index}].tool_call.arguments must be an object")
    return errors
