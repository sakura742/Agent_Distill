"""Run deterministic observable checks against the current Agent Runtime.

This first runner deliberately measures routing and workflow structure without
requiring an LLM or network API. Retrieval/answer benchmarks remain separate
because they require a populated local Chroma store and a model under test.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from agent.router import HybridRouter
from .benchmark import load_jsonl, accuracy


def run_routing(rows):
    router=HybridRouter()
    predictions=[]; gold=[]
    details=[]
    for r in rows:
        result=router.route(r["question"])
        predictions.append(result.domain); gold.append(r["gold"])
        details.append({"id":r["id"],"prediction":result.domain,"gold":r["gold"],"confidence":result.confidence,"method":result.method,"correct":result.domain==r["gold"]})
    return {"routing_accuracy":accuracy(predictions,gold),"details":details}


def main():
    p=argparse.ArgumentParser(); p.add_argument("path",default="data/evaluation/benchmark.jsonl",nargs="?"); p.add_argument("--output"); args=p.parse_args()
    rows=load_jsonl(args.path)
    routing=[r for r in rows if r.get("category")=="routing"]
    result={"benchmark":str(args.path),"samples":len(rows),"routing":run_routing(routing) if routing else {"routing_accuracy":0.0,"details":[]}}
    text=json.dumps(result,ensure_ascii=False,indent=2)
    print(text)
    if args.output: Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(text+"\n",encoding="utf-8")
if __name__ == "__main__": main()
