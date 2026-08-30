"""Gold benchmark for open-set legal domain routing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any
from agent.router import HybridRouter

def _load(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line)
            if not row.get("question") or not row.get("gold"):
                raise ValueError(f"{path}:{line_no} 缺少 question/gold")
            rows.append(row)
    return rows

def evaluate(rows: list[dict[str,Any]]) -> dict[str,Any]:
    router=HybridRouter(); details=[]; correct=0; by_class=Counter()
    for row in rows:
        result=router.route(row["question"]); pred=result.domain; gold=row["gold"]
        ok=pred==gold; correct+=ok; by_class[(gold,"total")]+=1; by_class[(gold,"correct")]+=int(ok)
        details.append({"id":row.get("id",row["question"]),"category":row.get("category"),"question":row["question"],"gold":gold,"prediction":pred,"confidence":result.confidence,"method":result.method,"candidates":result.candidates,"correct":ok})
    classes=sorted({r["gold"] for r in rows})
    per_class={c:{"samples":by_class[(c,"total")],"accuracy":by_class[(c,"correct")]/by_class[(c,"total")] if by_class[(c,"total")] else 0.0} for c in classes}
    return {"samples":len(rows),"accuracy":correct/len(rows) if rows else 0.0,"per_class":per_class,"details":details}

def main():
    p=argparse.ArgumentParser(); p.add_argument("benchmark",type=Path); p.add_argument("--output",type=Path,default=None); a=p.parse_args()
    report=evaluate(_load(a.benchmark)); text=json.dumps(report,ensure_ascii=False,indent=2)
    if a.output: a.output.write_text(text+"\n",encoding="utf-8")
    else: print(text)
if __name__=="__main__": main()
