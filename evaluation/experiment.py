"""Paired Phase 6 experiment runner.

The primary comparison is intentionally limited to Qwen3.5-4B Raw vs
Qwen3.5-4B LoRA. Model adapters are injected through the same callable
interface so benchmark execution remains identical.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from .benchmark import load_jsonl, evaluate, evaluate_citations

@dataclass(frozen=True)
class ModelSpec:
    name: str
    runner: Callable[[dict[str, Any]], dict[str, Any]]

def run_model(spec: ModelSpec, records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [spec.runner(r) for r in records]
    merged=[]
    for gold, pred in zip(records, predictions):
        row=dict(gold)
        row.update(pred)
        merged.append(row)
    result=evaluate(merged)
    result.update(evaluate_citations([r for r in merged if r.get("category")=="answer_citation"]))
    return {"model":spec.name,"samples":len(records),"metrics":result,"predictions":merged}

def paired_delta(raw: dict[str, Any], lora: dict[str, Any]) -> dict[str, float]:
    keys=set(raw["metrics"]) & set(lora["metrics"])
    return {k: float(lora["metrics"][k])-float(raw["metrics"][k]) for k in sorted(keys)}

def save_report(raw: dict[str, Any], lora: dict[str, Any], path: str | Path) -> None:
    report={"comparison":"Qwen3.5-4B Raw vs Qwen3.5-4B LoRA","raw":raw,"lora":lora,"delta_lora_minus_raw":paired_delta(raw,lora)}
    Path(path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
