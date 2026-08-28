"""End-to-end Phase 6 runner: Qwen3.5-4B Raw vs Qwen3.5-4B LoRA.

Only answer/citation outputs depend on the model in the current Phase 4 graph;
routing/tool execution remain the same Runtime components. The runner keeps
those deterministic metrics and model-dependent generation metrics separate.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from agent.runtime.graph import build_legal_agent_graph
from agent.runtime.qwen35_generator import Qwen35AnswerGenerator
from model_service.qwen35 import Qwen35Service
from configs.settings import settings
from .benchmark import load_jsonl
from .metrics import token_overlap_f1, citation_precision, citation_recall


def run(model_name: str, adapter: str | None, rows: list[dict]) -> dict:
    service = Qwen35Service(adapter_path=adapter)
    graph = build_legal_agent_graph(answer_generator=Qwen35AnswerGenerator(service))
    outputs=[]
    for row in rows:
        if row.get("category") not in {"answer", "answer_citation", "routing", "tool_calling", "workflow"}: continue
        if "question" not in row: continue
        started=time.perf_counter()
        state=graph.invoke({"question": row["question"]})
        outputs.append({"id":row["id"],"category":row["category"],"question":row["question"],"gold":row.get("gold"),"prediction":state.get("answer",""),"domain":state.get("domain"),"tool_name":state.get("tool_name"),"citations":state.get("citations",[]),"verification":state.get("verification",{}),"trace":state.get("trace",[]),"latency_ms":round((time.perf_counter()-started)*1000,2)})
    answer_rows=[x for x in outputs if x["category"]=="answer"]
    citation_rows=[x for x in outputs if x["category"]=="answer_citation"]
    metrics={"answer_token_overlap_f1":sum(token_overlap_f1(x["prediction"],x["gold"] or "") for x in answer_rows)/len(answer_rows) if answer_rows else 0.0,
             "citation_precision":sum(citation_precision([c.get("reference") for c in x["citations"]],x["gold"] or []) for x in citation_rows)/len(citation_rows) if citation_rows else 0.0,
             "citation_recall":sum(citation_recall([c.get("reference") for c in x["citations"]],x["gold"] or []) for x in citation_rows)/len(citation_rows) if citation_rows else 0.0,
             "avg_latency_ms":sum(x["latency_ms"] for x in outputs)/len(outputs) if outputs else 0.0}
    service.unload()
    return {"model":model_name,"metrics":metrics,"predictions":outputs}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--benchmark",default="data/evaluation/benchmark.jsonl"); p.add_argument("--raw-output",default="data/evaluation/results/qwen35_4b_raw.json"); p.add_argument("--lora-output",default="data/evaluation/results/qwen35_4b_lora.json"); args=p.parse_args()
    rows=load_jsonl(args.benchmark)
    raw=run("Qwen3.5-4B Raw",None,rows)
    lora=run("Qwen3.5-4B LoRA",str(settings.qwen35_lora_output_dir),rows)
    Path(args.raw_output).parent.mkdir(parents=True,exist_ok=True); Path(args.lora_output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.raw_output).write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding="utf-8")
    Path(args.lora_output).write_text(json.dumps(lora,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"raw":raw["metrics"],"lora":lora["metrics"]},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
