# Phase 6 Evaluation Report

## Experiment

The primary comparison is strictly paired:

- **Qwen3.5-4B Raw** — base checkpoint without the Phase 5 adapter.
- **Qwen3.5-4B LoRA** — the Phase 5 LoRA checkpoint.

Both variants use the same benchmark, legal corpus, LangGraph Runtime, MCP/tool contracts, generation settings and evaluation code. This isolates the effect of the LoRA adaptation.

## Evaluation dimensions

| Dimension | Metrics |
|---|---|
| Routing | Routing Accuracy |
| Retrieval | Recall@5, MRR |
| Tool Calling | Tool Selection Accuracy, Argument Accuracy |
| Workflow | Workflow Success Rate, Interruption Error Rate |
| Answer | Token-overlap F1 |
| Citation | Citation Precision, Recall, Accuracy |
| Runtime | Average latency, runtime error rate |

The benchmark records observable state and execution traces rather than relying on hidden chain-of-thought.

## Error-analysis loop

1. Run identical benchmark against Raw and LoRA.
2. Classify failures into routing, retrieval, tool selection, tool parameters, workflow, verification, citation, runtime and retry/interruption categories.
3. Rank failed samples by number of error types and export them as hard examples.
4. Feed selected hard examples into the Phase 5 trajectory/distillation training set.
5. Re-train LoRA and rerun the same benchmark.
6. Compare paired metrics and interruption/error-rate reduction.

The automation entry point is:

```bash
python -m evaluation.run_iteration \
  --raw data/evaluation/results/qwen35_4b_raw.json \
  --lora data/evaluation/results/qwen35_4b_lora.json
```

It writes `data/evaluation/iteration_report.json` and the hard-example export under `distill/data/`.

## 35% claim policy

The report calculates relative reduction as:

`(Raw interruption/error rate - LoRA interruption/error rate) / Raw interruption/error rate`

The `target_35_percent_met` field is true only when the measured reduction is at least 35%. No experimental number is hard-coded into the project documentation.

Therefore, the resume claim **“将长链条任务的中断率/错误率降低 35%”** should only be used after a real paired run produces `target_35_percent_met=true` on a sufficiently large held-out benchmark.
