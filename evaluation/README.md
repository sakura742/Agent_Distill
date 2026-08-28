# Phase 6 Benchmark / Evaluation

## Benchmark

当前主 Benchmark 共 **40 个法律场景**，覆盖：

- Routing：10
- Retrieval：6
- Tool Calling：6
- Workflow：6
- Answer：6
- Citation：6

主对照严格限定为 **Qwen3.5-4B Raw vs Qwen3.5-4B LoRA**，两者共享 Runtime、RAG、Tool Contract、Benchmark 与生成配置。

## Metrics

- Retrieval：Recall@5、MRR
- Tool：Tool Selection Accuracy、Parameter Accuracy
- Workflow：Workflow Success Rate、Interruption Error Rate
- Answer：Token-overlap F1
- Citation：Precision、Recall、Exact Accuracy
- System：Runtime Error Rate、Average Latency

所有指标均由实际逐样本输出计算，不预填结果。

## Error Analysis / Iteration

`evaluation.error_analysis` 对失败样本进行分类：routing、retrieval、tool selection、tool parameter、workflow、citation、runtime、verification、retry/interruption 等，并按错误数量生成 hard score。

`evaluation.iteration` 将 Raw / LoRA 结果做 paired comparison，输出指标增量、错误分布以及相对错误率下降；其中 **35% 是实验验收目标，不是预设结果**。

```bash
python -m evaluation.run_qwen35_benchmark
python -m evaluation.iteration \
  data/evaluation/results/qwen35_4b_raw.json \
  data/evaluation/results/qwen35_4b_lora.json
```

输出：

```text
data/evaluation/results/qwen35_4b_raw.json
data/evaluation/results/qwen35_4b_lora.json
data/evaluation/iteration_report.json
distill/data/evaluation_hard_examples.jsonl
```
