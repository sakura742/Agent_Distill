"""End-to-end Phase 6 benchmark: Qwen3.5-4B Raw vs Qwen3.5-4B LoRA.

The two models run through the same LangGraph Runtime, RAG corpus, tool
contracts and benchmark.  Metrics are computed from observable state/trace,
not from hidden reasoning.
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
    accuracy,
    argument_accuracy,
    citation_accuracy,
    citation_precision,
    citation_recall,
    interruption_error,
    mrr,
    recall_at_k,
    token_overlap_f1,
    workflow_success,
)

EVAL_CATEGORIES = {
    "routing",
    "retrieval",
    "tool_calling",
    "workflow",
    "answer",
    "answer_citation",
}


def _refs(state: dict[str, Any]) -> list[str]:
    docs = state.get("retrieved_documents") or state.get("citations") or []
    return [d.get("reference", "") for d in docs if isinstance(d, dict) and d.get("reference")]


def run(model_name: str, adapter: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    service = Qwen35Service(adapter_path=adapter)
    graph = build_legal_agent_graph(answer_generator=Qwen35AnswerGenerator(service))
    outputs: list[dict[str, Any]] = []

    for row in rows:
        category = row.get("category")
        if category not in EVAL_CATEGORIES or "question" not in row:
            continue
        started = time.perf_counter()
        state = graph.invoke({"question": row["question"]})
        latency = round((time.perf_counter() - started) * 1000, 2)
        outputs.append({
            "id": row["id"],
            "category": category,
            "question": row["question"],
            "gold": row.get("gold"),
            "prediction": state.get("answer", ""),
            "route_prediction": state.get("domain"),
            "tool_prediction": {
                "name": state.get("tool_name"),
                "arguments": state.get("tool_arguments", {}),
            },
            "retrieved": _refs(state),
            "citations": _refs(state),
            "verification": state.get("verification", {}),
            "trace": state.get("trace", []),
            "retry_count": state.get("retry_count", 0),
            "error": state.get("error"),
            "latency_ms": latency,
        })

    def rows_of(category: str) -> list[dict[str, Any]]:
        return [x for x in outputs if x["category"] == category]

    routing = rows_of("routing")
    retrieval = rows_of("retrieval")
    tools = rows_of("tool_calling")
    workflows = rows_of("workflow")
    answers = rows_of("answer")
    citations = rows_of("answer_citation")

    metrics: dict[str, float] = {}
    if routing:
        metrics["routing_accuracy"] = accuracy(
            [x["route_prediction"] for x in routing],
            [x["gold"] for x in routing],
        )
    if retrieval:
        metrics["retrieval_recall_at_5"] = recall_at_k(
            [x["retrieved"] for x in retrieval],
            [x["gold"] for x in retrieval],
            5,
        )
        metrics["retrieval_mrr"] = mrr(
            [x["retrieved"] for x in retrieval],
            [x["gold"] for x in retrieval],
        )
    if tools:
        metrics["tool_selection_accuracy"] = accuracy(
            [x["tool_prediction"]["name"] for x in tools],
            [x["gold"]["name"] for x in tools],
        )
        metrics["tool_argument_accuracy"] = sum(
            argument_accuracy(x["tool_prediction"]["arguments"], x["gold"].get("arguments", {}))
            for x in tools
        ) / len(tools)
    if workflows:
        metrics["workflow_success_rate"] = sum(
            workflow_success(x["trace"], x["verification"]) for x in workflows
        ) / len(workflows)
        metrics["interruption_error_rate"] = sum(
            interruption_error(x["trace"], x["verification"]) for x in workflows
        ) / len(workflows)
    if answers:
        metrics["answer_token_overlap_f1"] = sum(
            token_overlap_f1(x["prediction"], x["gold"] or "") for x in answers
        ) / len(answers)
    if citations:
        metrics["citation_precision"] = sum(
            citation_precision(x["citations"], x["gold"] or []) for x in citations
        ) / len(citations)
        metrics["citation_recall"] = sum(
            citation_recall(x["citations"], x["gold"] or []) for x in citations
        ) / len(citations)
        metrics["citation_accuracy"] = sum(
            citation_accuracy(x["citations"], x["gold"] or []) for x in citations
        ) / len(citations)
    metrics["avg_latency_ms"] = (
        sum(x["latency_ms"] for x in outputs) / len(outputs) if outputs else 0.0
    )
    metrics["error_rate"] = (
        sum(bool(x.get("error")) for x in outputs) / len(outputs) if outputs else 0.0
    )

    service.unload()
    return {"model": model_name, "metrics": metrics, "predictions": outputs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/evaluation/benchmark.jsonl")
    p.add_argument("--raw-output", default="data/evaluation/results/qwen35_4b_raw.json")
    p.add_argument("--lora-output", default="data/evaluation/results/qwen35_4b_lora.json")
    args = p.parse_args()

    rows = load_jsonl(args.benchmark)
    raw = run("Qwen3.5-4B Raw", None, rows)
    lora = run("Qwen3.5-4B LoRA", str(settings.qwen35_lora_output_dir), rows)

    for path, result in ((args.raw_output, raw), (args.lora_output, lora)):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"raw": raw["metrics"], "lora": lora["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
