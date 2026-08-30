import json

def load(path):
    return json.load(open(path, encoding="utf-8"))["predictions"]

rows = load("data/evaluation/results/tmp_retrieval.json") + load("data/evaluation/results/tmp_citation.json")

for r in rows:
    print("=" * 60)
    print(f"[{r['id']}] {r['question']}")
    print(f"当前 gold: {r['gold']}")
    details = r.get("citation_details") or []
    if not details:
        print("  (citation_details 为空，可能这条 case 检索没结果)")
    for i, c in enumerate(details, 1):
        ref = c.get("reference", "")
        content = (c.get("content") or "").replace("\n", " ")[:120]
        print(f"  候选{i}: {ref}")
        print(f"    正文: {content}")
    print()