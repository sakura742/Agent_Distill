"""Adapter from the LangGraph generation node to Qwen3.5 serving."""

from __future__ import annotations

from typing import Any

from model_service.qwen35 import GenerationConfig, Qwen35Service


class Qwen35AnswerGenerator:
    def __init__(self, service: Qwen35Service | None = None) -> None:
        self.service = service or Qwen35Service()

    def __call__(self, question: str, evidence: str, citations: list[dict[str, Any]]) -> str:
        return self.service.generate(
            [
                {
                    "role": "system",
                    "content": "你是法律 Agent。只能依据提供的检索证据回答；证据不足时明确说明。",
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n法律证据：\n{evidence}",
                },
            ],
            GenerationConfig(max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=False),
        )
