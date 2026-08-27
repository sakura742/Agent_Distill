"""Dataset schema and benchmark runner for observable Agent behavior."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .metrics import accuracy, recall_at_k, mrr, citation_precision, citation_recall, token_overlap_f1

CATEGORIES = ("routing", "retrieval", "tool_calling", "workflow", "answer", "multi_turn")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by={c:[] for c in CATEGORIES}
    for r in records:
        by.setdefault(r.get("category", "answer"), []).append(r)
    out={}
    routing=by["routing"]
    if routing: out["routing_accuracy"]=accuracy([r.get("prediction") for r in routing],[r.get("gold") for r in routing])
    retrieval=by["retrieval"]
    if retrieval:
        out["retrieval_recall_at_5"]=recall_at_k([r.get("retrieved",[]) for r in retrieval],[r.get("relevant",[]) for r in retrieval],5)
        out["retrieval_mrr"]=mrr([r.get("retrieved",[]) for r in retrieval],[r.get("relevant",[]) for r in retrieval])
    tools=by["tool_calling"]
    if tools:
        out["tool_selection_accuracy"]=accuracy([r.get("prediction",{}).get("name") for r in tools],[r.get("gold",{}).get("name") for r in tools])
        out["tool_argument_accuracy"]=accuracy([json.dumps(r.get("prediction",{}).get("arguments",{}),sort_keys=True,ensure_ascii=False) for r in tools],[json.dumps(r.get("gold",{}).get("arguments",{}),sort_keys=True,ensure_ascii=False) for r in tools])
    workflows=by["workflow"]
    if workflows: out["workflow_success_rate"]=accuracy([r.get("prediction") for r in workflows],[r.get("gold") for r in workflows])
    answers=by["answer"]
    if answers: out["answer_token_overlap_f1"]=sum(token_overlap_f1(r.get("prediction",""),r.get("gold","")) for r in answers)/len(answers)
    multi=by["multi_turn"]
    if multi: out["multi_turn_success_rate"]=accuracy([r.get("prediction") for r in multi],[r.get("gold") for r in multi])
    return out


def evaluate_citations(records: list[dict[str, Any]]) -> dict[str,float]:
    if not records: return {"citation_precision":0.0,"citation_recall":0.0}
    ps=[]; rs=[]
    for r in records:
        ps.append(citation_precision(r.get("predicted",[]),r.get("gold",[])))
        rs.append(citation_recall(r.get("predicted",[]),r.get("gold",[])))
    return {"citation_precision":sum(ps)/len(ps),"citation_recall":sum(rs)/len(rs)}
