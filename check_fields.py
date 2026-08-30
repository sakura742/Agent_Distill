import json

raw = json.load(open("data/evaluation/results/qwen35_4b_raw.json", encoding="utf-8"))
row = next(r for r in raw["predictions"] if r["category"] == "answer_citation")
print("这一行有哪些字段:", list(row.keys()))
print("citation_details 内容:", row.get("citation_details"))