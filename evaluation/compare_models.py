"""CLI for the controlled Qwen3.5-4B Raw/LoRA comparison.

The actual model runners are supplied by the local environment; this script
only guarantees that both runners consume the exact same benchmark rows.
"""
from __future__ import annotations
import argparse
from .benchmark import load_jsonl
from .experiment import ModelSpec, run_model, save_report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("benchmark")
    p.add_argument("report")
    p.add_argument("--raw-runner", required=True, help="module:function")
    p.add_argument("--lora-runner", required=True, help="module:function")
    args=p.parse_args()
    import importlib
    def load(spec):
        module, fn=spec.split(":",1)
        return getattr(importlib.import_module(module),fn)
    records=load_jsonl(args.benchmark)
    raw=run_model(ModelSpec("Qwen3.5-4B Raw",load(args.raw_runner)),records)
    lora=run_model(ModelSpec("Qwen3.5-4B LoRA",load(args.lora_runner)),records)
    save_report(raw,lora,args.report)
    print(f"saved: {args.report}")

if __name__ == "__main__": main()
