"""Convert canonical trajectories into the SFT JSONL format consumed by training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            steps = []
            for step in row.get("steps", []):
                item = f"步骤 {step['step']}：{step['action']}"
                if step.get("tool_call"):
                    item += "\n工具调用：" + json.dumps(step["tool_call"], ensure_ascii=False)
                if step.get("observation") is not None:
                    item += "\n工具结果：" + json.dumps(step["observation"], ensure_ascii=False)
                steps.append(item)
            output = "\n".join([
                f"意图：{row['intent']}",
                f"法域：{row.get('domain') or 'none'}",
                *steps,
                f"最终回答：{row.get('final_answer') or ''}",
            ])
            dst.write(json.dumps({
                "instruction": "分析法律问题，按需选择正确工具并完成多步骤任务。",
                "input": row["user_query"],
                "output": output,
                "metadata": row.get("metadata", {}),
            }, ensure_ascii=False) + "\n")
            written += 1
    return total, written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    total, written = convert(args.input, args.output)
    print(f"read={total} written={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
