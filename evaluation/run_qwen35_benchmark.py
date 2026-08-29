"""End-to-end Phase 6 benchmark: Qwen3.5-4B Raw vs Qwen3.5-4B LoRA.

Both variants use the same LangGraph Runtime, RAG corpus, tool contracts and
benchmark. Metrics are computed from observable state/trace.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agent.runtime.graph import build_legal_agent_graph
from agent.runtime.qwen35_generator import Qwen35AnswerGenerator
from configs.settings import settings
from model_service.qwen35 import Qwen35Service

from .benchmark import load_jsonl
from .metrics import (
    accuracy, argument_accuracy, citation_accuracy, citation_precision,
    citation_recall, interruption_error, mrr, recall_at_k,
    token_overlap_f1, workflow_success,
)

EVAL_CATEGORIES = {"routing", "retrieval", "tool_calling", "workflow", "answer", "answer_citation"}


def _refs(state: dict[str, Any]) -> list[str]:
    docs = state.get("retrieved_documents") or state.get("citations") or []
    return [d.get("reference", "") for d in docs if isinstance(d, dict) and d.get("reference")]


def run(model_name: str, model_path: str, adapter: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Not a HuggingFace model directory (config.json missing): {model_dir}")
    if adapter:
        adapter_dir = Path(adapter)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_dir}")
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"Not a PEFT adapter directory (adapter_config.json missing): {adapter_dir}")

    service = Qwen35Service(model_path=str(model_dir), adapter_path=adapter)
    graph = build_legal_agent_graph(answer_generator=Qwen35AnswerGenerator(service))
    outputs: list[dict[str, Any]] = []

    for index, row in enumerate(rows, 1):
        category = row.get("category")
        if category not in EVAL_CATEGORIES or "question" not in row:
            continue
        print(f"[{model_name}] {index}/{len(rows)} {row['id']}", flush=True)
        started = time.perf_counter()
        try:
            state = graph.invoke({"question": row["question"]})
            error = state.get("error")
        except Exception as exc:
            state = {}
            error = f"{type(exc).__name__}: {exc}"
        latency = round((time.perf_counter() - started) * 1000, 2)
        outputs.append({
            "id": row["id"], "category": category, "question": row["question"],
            "gold": row.get("gold"), "prediction": state.get("answer", ""),
            "route_prediction": state.get("domain"),
            "tool_prediction": {"name": state.get("tool_name"), "arguments": state.get("tool_arguments", {})},
            "retrieved": _refs(state), "citations": _refs(state),
            "verification": state.get("verification", {}), "trace": state.get("trace", []),
            "retry_count": state.get("retry_count", 0), "error": error, "latency_ms": latency,
        })

    def rows_of(category: str) -> list[dict[str, Any]]:
        return [x for x in outputs if x["category"] == category]

    routing, retrieval = rows_of("routing"), rows_of("retrieval")
    tools, workflows = rows_of("tool_calling"), rows_of("workflow")
    answers, citations = rows_of("answer"), rows_of("answer_citation")
    metrics: dict[str, float] = {}
    if routing:
        metrics["routing_accuracy"] = accuracy([x["route_prediction"] for x in routing], [x["gold"] for x in routing])
    if retrieval:
        metrics["retrieval_recall_at_5"] = recall_at_k([x["retrieved"] for x in retrieval], [x["gold"] for x in retrieval], 5)
        metrics["retrieval_mrr"] = mrr([x["retrieved"] for x in retrieval], [x["gold"] for x in retrieval])
    if tools:
        metrics["tool_selection_accuracy"] = accuracy([x["tool_prediction"]["name"] for x in tools], [x["gold"]["name"] for x in tools])
        metrics["tool_argument_accuracy"] = sum(argument_accuracy(x["tool_prediction"]["arguments"], x["gold"].get("arguments", {})) for x in tools) / len(tools)
    if workflows:
        metrics["workflow_success_rate"] = sum(workflow_success(x["trace"], x["verification"]) for x in workflows) / len(workflows)
        metrics["interruption_error_rate"] = sum(interruption_error(x["trace"], x["verification"]) for x in workflows) / len(workflows)
    if answers:
        metrics["answer_token_overlap_f1"] = sum(token_overlap_f1(x["prediction"], x["gold"] or "") for x in answers) / len(answers)
    if citations:
        metrics["citation_precision"] = sum(citation_precision(x["citations"], x["gold"] or []) for x in citations) / len(citations)
        metrics["citation_recall"] = sum(citation_recall(x["citations"], x["gold"] or []) for x in citations) / len(citations)
        metrics["citation_accuracy"] = sum(citation_accuracy(x["citations"], x["gold"] or []) for x in citations) / len(citations)
    metrics["avg_latency_ms"] = sum(x["latency_ms"] for x in outputs) / len(outputs) if outputs else 0.0
    metrics["error_rate"] = sum(bool(x.get("error")) for x in outputs) / len(outputs) if outputs else 0.0
    service.unload()
    return {"model": model_name, "model_path": str(model_dir), "adapter_path": adapter, "metrics": metrics, "predictions": outputs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/evaluation/benchmark.jsonl")
    p.add_argument("--model-path", default=settings.qwen35_model_path)
    p.add_argument("--lora-path", default=str(settings.qwen35_lora_output_dir))
    p.add_argument("--raw-output", default="data/evaluation/results/qwen35_4b_raw.json")
    p.add_argument("--lora-output", default="data/evaluation/results/qwen35_4b_lora.json")
    args = p.parse_args()

    rows = load_jsonl(args.benchmark)
    raw = run("Qwen3.5-4B Raw", args.model_path, None, rows)
    lora = run("Qwen3.5-4B LoRA", args.model_path, args.lora_path, rows)
    for path, result in ((args.raw_output, raw), (args.lora_output, lora)):
        output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"raw": raw["metrics"], "lora": lora["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
