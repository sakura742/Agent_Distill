import json

raw = json.load(open("data/evaluation/results/qwen35_4b_raw.json", encoding="utf-8"))
rows = [r for r in raw["predictions"] if r["category"] in ("retrieval", "answer_citation")]

for r in rows:
    print(f"[{r['id']}] {r['question']}")
    print(f"  当前 gold: {r['gold']}")
    for i, c in enumerate(r.get("citation_details") or [], 1):
        print(f"  候选{i}: {c['reference']}")
        print(f"    正文: {c['content'][:150]}")
    print()