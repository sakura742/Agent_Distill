"""Build Phase-6 benchmark records from a hand-curated gold JSONL.

Trajectory files are intentionally *not* treated as gold labels: using a model's
own retrieval/output as reference would leak evaluation targets. This script
only validates/normalizes curated cases and can optionally attach model outputs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

CATEGORIES = {"routing", "retrieval", "tool_calling", "workflow", "answer", "answer_citation", "multi_turn"}
REQUIRED = {
    "routing": {"question", "gold"},
    "retrieval": {"question", "relevant"},
    "tool_calling": {"question", "gold"},
    "workflow": {"question", "gold"},
    "answer": {"question", "gold"},
    "answer_citation": {"question", "gold"},
    "multi_turn": {"conversation", "gold"},
}

def validate(rows):
    errors=[]
    for i,r in enumerate(rows,1):
        c=r.get("category")
        if c not in CATEGORIES: errors.append(f"line {i}: invalid category {c!r}"); continue
        missing=REQUIRED[c]-r.keys()
        if missing: errors.append(f"line {i}: missing {sorted(missing)}")
        if "id" not in r: errors.append(f"line {i}: missing id")
    return errors

def main():
    p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("output"); args=p.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    errors=validate(rows)
    if errors: raise SystemExit("\n".join(errors))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
    counts={c:sum(r["category"]==c for r in rows) for c in sorted(CATEGORIES)}
    print(json.dumps({"samples":len(rows),"categories":counts,"output":args.output},ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
