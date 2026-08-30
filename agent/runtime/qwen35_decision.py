"""Qwen3.5 decision generator used by the Agent Runtime."""
from __future__ import annotations
import json
from typing import Any
from model_service.qwen35 import GenerationConfig, Qwen35Service
SYSTEM="你是法律 Agent。根据用户问题选择法域、意图和合适的法律检索工具。只输出合法 JSON，不要解释。"
class Qwen35DecisionGenerator:
    def __init__(self, service: Qwen35Service | None=None): self.service=service or Qwen35Service()
    def __call__(self, question: str) -> dict[str,Any]|None:
        text=self.service.generate([{"role":"system","content":SYSTEM},{"role":"user","content":question}],GenerationConfig(max_new_tokens=256,temperature=0.0,top_p=1.0,do_sample=False))
        try:
            start=text.find("{"); end=text.rfind("}")
            if start<0 or end<=start:return None
            obj=json.loads(text[start:end+1]); tool=obj.get("tool") or {}
            if not obj.get("domain") or not tool.get("name"):return None
            return obj
        except (json.JSONDecodeError,TypeError,AttributeError): return None
