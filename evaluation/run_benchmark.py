"""CLI: python -m evaluation.run_benchmark evaluation/benchmark.jsonl"""
from __future__ import annotations
import argparse, json
from .benchmark import load_jsonl, evaluate, evaluate_citations

def main():
    p=argparse.ArgumentParser(); p.add_argument("path"); args=p.parse_args()
    rows=load_jsonl(args.path)
    result=evaluate(rows)
    result.update(evaluate_citations([r for r in rows if r.get("category") == "answer_citation"]))
    print(json.dumps({"samples":len(rows),"metrics":result},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
