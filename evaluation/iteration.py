"""One-command evaluation -> error analysis -> hard-example mining loop."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .error_analysis import analyze, write_hard_examples


def build_iteration_report(raw_result: str, lora_result: str, output: str) -> None:
    raw=json.loads(Path(raw_result).read_text(encoding="utf-8")); lora=json.loads(Path(lora_result).read_text(encoding="utf-8"))
    raw_a=analyze(raw["predictions"]); lora_a=analyze(lora["predictions"])
    metrics=sorted(set(raw["metrics"]) & set(lora["metrics"]))
    delta={m:lora["metrics"][m]-raw["metrics"][m] for m in metrics}
    report={"comparison":"Qwen3.5-4B Raw vs Qwen3.5-4B LoRA","raw_metrics":raw["metrics"],"lora_metrics":lora["metrics"],"delta_lora_minus_raw":delta,"raw_errors":raw_a["error_counts"],"lora_errors":lora_a["error_counts"]}
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("raw"); p.add_argument("lora"); p.add_argument("--report",default="data/evaluation/iteration_report.json"); p.add_argument("--hard-output",default="distill/data/evaluation_hard_examples.jsonl"); a=p.parse_args()
    build_iteration_report(a.raw,a.lora,a.report)
    lora=json.loads(Path(a.lora).read_text(encoding="utf-8")); n=write_hard_examples(lora["predictions"],a.hard_output); print(f"hard examples: {n}")

if __name__=="__main__": main()
