"""Paired evaluation -> error analysis -> hard-example mining report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .error_analysis import analyze

# routing / retrieval / tool_calling 三类不经过被比较的 Qwen 模型（见
# agent/runtime/nodes.py：intent_analysis 用 agent/router.py 的
# HybridRouter 规则+embedding 路由，tool_decision 是纯字符串拼接，两者都
# 不调用 answer_generator）。Raw 和 LoRA 在这三类上的输出在代码层面就是
# 完全确定性、逐字节相同的，把它们计入"Raw vs LoRA 错误率对比"只会用
# 跟模型无关的失败稀释掉真实差距，甚至完全掩盖掉。
# 只有 workflow / answer / answer_citation 这三类的 answer 文本真正经过
# generation 节点、调用了被比较的模型，是唯一能反映 Raw/LoRA 差距的子集。
_MODEL_DEPENDENT_CATEGORIES = {"workflow", "answer", "answer_citation"}


def _by_category(rows: list[dict[str, Any]], categories: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("category") in categories]


def relative_reduction(before: float, after: float) -> float | None:
    if before <= 0:
        return None
    return (before - after) / before


def _paired(raw_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_id = {x.get("id"): x for x in raw_rows}
    lora_by_id = {x.get("id"): x for x in lora_rows}
    result = []
    for case_id in sorted(set(raw_by_id) | set(lora_by_id)):
        r, l = raw_by_id.get(case_id), lora_by_id.get(case_id)
        if r is None or l is None:
            continue
        result.append({
            "id": case_id,
            "category": r.get("category", l.get("category")),
            "raw_error": bool(r.get("error")),
            "lora_error": bool(l.get("error")),
            "raw_retry_count": r.get("retry_count", 0),
            "lora_retry_count": l.get("retry_count", 0),
            "raw_prediction": r.get("prediction"),
            "lora_prediction": l.get("prediction"),
        })
    return result


def _hard_examples(raw_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_id = {x.get("id"): x for x in raw_rows}
    lora_by_id = {x.get("id"): x for x in lora_rows}
    items = []
    for case_id in sorted(set(raw_by_id) | set(lora_by_id)):
        r, l = raw_by_id.get(case_id), lora_by_id.get(case_id)
        if not r or not l:
            continue
        re = analyze([r])["failed_samples"] > 0
        le = analyze([l])["failed_samples"] > 0
        if not (re or le):
            continue
        item = dict(l)
        item["hard_example_reason"] = (
            "raw_failed_lora_fixed" if re and not le else
            "both_failed" if re and le else "lora_regression"
        )
        item["raw_error_types"] = analyze([r])["failed_samples"] and analyze([r])["hard_examples"][0].get("error_types", [])
        item["lora_error_types"] = analyze([l])["failed_samples"] and analyze([l])["hard_examples"][0].get("error_types", [])
        items.append(item)
    return items


def build_iteration_report(raw_result: str, lora_result: str, output: str, hard_output: str | None = None) -> dict[str, Any]:
    raw = json.loads(Path(raw_result).read_text(encoding="utf-8"))
    lora = json.loads(Path(lora_result).read_text(encoding="utf-8"))
    raw_a, lora_a = analyze(raw["predictions"]), analyze(lora["predictions"])

    raw_metrics, lora_metrics = raw["metrics"], lora["metrics"]
    common = sorted(set(raw_metrics) & set(lora_metrics))
    delta = {m: lora_metrics[m] - raw_metrics[m] for m in common}
    # 之前这里读的是 metrics["interruption_error_rate"]（只覆盖 40 条里的 6 条
    # workflow case，且判定标准只看"回答非空+有引用"，跟答案对不对无关），一旦
    # 该 key 不存在还会静默 fallback 到 metrics["error_rate"]（纯粹的 Runtime
    # 崩溃率）。两者都不能反映 Raw/LoRA 的真实任务质量差距。
    #
    # 这里改用按 gold 标签逐类别判定正确性的口径（见
    # evaluation/error_analysis.classify），但只统计 _MODEL_DEPENDENT_CATEGORIES
    # （workflow/answer/answer_citation），排除 routing/retrieval/tool_calling——
    # 后三者是规则/embedding/RAG 检索出的确定性结果，不经过被比较的 Qwen
    # 模型，Raw 和 LoRA 在这三类上的表现必然完全相同，混进来只会稀释或
    # 掩盖模型本身的真实差距。
    raw_model_rows = _by_category(raw["predictions"], _MODEL_DEPENDENT_CATEGORIES)
    lora_model_rows = _by_category(lora["predictions"], _MODEL_DEPENDENT_CATEGORIES)
    raw_model_a, lora_model_a = analyze(raw_model_rows), analyze(lora_model_rows)
    raw_interrupt = raw_model_a["failed_samples"] / raw_model_a["samples"] if raw_model_a["samples"] else 0.0
    lora_interrupt = lora_model_a["failed_samples"] / lora_model_a["samples"] if lora_model_a["samples"] else 0.0
    reduction = relative_reduction(raw_interrupt, lora_interrupt)

    report = {
        "comparison": "Qwen3.5-4B Raw vs Qwen3.5-4B LoRA",
        "benchmark_samples": len(raw["predictions"]),
        "paired_samples": len(_paired(raw["predictions"], lora["predictions"])),
        "raw_metrics": raw_metrics,
        "lora_metrics": lora_metrics,
        "delta_lora_minus_raw": delta,
        "error_reduction": {
            "scope": sorted(_MODEL_DEPENDENT_CATEGORIES),
            "scope_note": (
                "仅统计 workflow/answer/answer_citation：routing/retrieval/"
                "tool_calling 由规则路由与 RAG 检索产生，不经过被比较的模型，"
                "Raw/LoRA 在这三类上的结果恒定相同，不计入对比。"
            ),
            "samples_used": raw_model_a["samples"],
            "baseline_error_rate": raw_interrupt,
            "lora_error_rate": lora_interrupt,
            "relative_reduction": reduction,
            "target_35_percent_met": reduction is not None and reduction >= 0.35,
        },
        "raw_errors": raw_a["error_counts"],
        "lora_errors": lora_a["error_counts"],
        "raw_failed_samples": raw_a["failed_samples"],
        "lora_failed_samples": lora_a["failed_samples"],
        "paired_cases": _paired(raw["predictions"], lora["predictions"]),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if hard_output:
        hp = Path(hard_output)
        hp.parent.mkdir(parents=True, exist_ok=True)
        with hp.open("w", encoding="utf-8") as f:
            for row in _hard_examples(raw["predictions"], lora["predictions"]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["hard_examples"] = len(_hard_examples(raw["predictions"], lora["predictions"]))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw")
    p.add_argument("lora")
    p.add_argument("--report", default="data/evaluation/iteration_report.json")
    p.add_argument("--hard-output", default="distill/data/evaluation_hard_examples.jsonl")
    a = p.parse_args()
    report = build_iteration_report(a.raw, a.lora, a.report, a.hard_output)
    print(json.dumps(report["error_reduction"], ensure_ascii=False, indent=2))
    print(f"hard examples: {report.get('hard_examples', 0)}")


if __name__ == "__main__":
    main()
