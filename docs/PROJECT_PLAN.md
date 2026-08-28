# Agent_Distill → Legal Agent Platform 项目计划

| 阶段 | 名称 | 状态 |
|---|---|---|
| 0 | 代码审计 | ✅ |
| 0.5 | 目标态规划 | ✅ |
| 1 | 模块化骨架 + 配置/日志/异常 | ✅ |
| 2 | MCP Tool Service + 工具契约 | ✅ |
| 3 | RAG 分域 + Metadata / Rerank | ✅ |
| 4 | LangGraph Runtime + Hybrid Router | ✅ |
| 5 | Qwen3.5-4B + Trajectory 蒸馏 | ✅ |
| 6 | Benchmark / Evaluation 体系化 | ✅ 实现完成 / 待本地实测 |
| 7 | 多轮对话 + Web Dashboard + Docker | ⬜ |

## Phase 6

主实验固定为 **Qwen3.5-4B Raw vs Qwen3.5-4B LoRA**。

Benchmark 共 **40 个法律场景**，覆盖 Routing、Retrieval、Tool Calling、Workflow、Answer、Citation。

指标覆盖：Recall@5、MRR、Tool Selection Accuracy、Tool Parameter Accuracy、Workflow Success Rate、最终回答 Token-overlap F1、Citation Precision / Recall / Exact Accuracy、平均端到端延迟、Workflow Interruption Error Rate、Runtime Error Rate。

### 已完成

- Benchmark schema 与 deterministic metrics。
- Raw / LoRA 共享 Runtime 的 paired benchmark runner。
- 错误分类：routing、retrieval、tool selection、tool parameter、workflow、verification、citation、runtime、retry/interruption。
- 困难样本挖掘，并区分 `raw_failed_lora_fixed`、`both_failed`、`lora_regression`。
- Paired case-by-case 对比与指标 delta。
- 35% 错误率下降自动判定，不硬编码实验结果。
- Benchmark 运行前数据校验。
- 最终实验结果校验 gate。

### 实验闭环

`Benchmark → Raw/LoRA paired evaluation → Error Analysis → Hard Example Mining → LoRA retraining → Re-evaluation`

代码入口：

```bash
# 1. 运行相同 Benchmark
python -m evaluation.run_qwen35_benchmark

# 2. 生成 paired 对比、错误分析与 hard examples
python -m evaluation.run_iteration

# 3. 最终结果校验
python -m evaluation.finalize_phase6
```

### 简历指标口径

长链条任务中断率/错误率下降定义为：

`(Raw interruption_error_rate - LoRA interruption_error_rate) / Raw interruption_error_rate`

只有最终校验结果 `target_35_percent_met=true` 时，才能在简历中写“降低 35%”；否则使用实际测量值。

### 实测边界

代码层面的 Phase 6 已完成。Qwen3.5-4B 的真实全量推理必须在具备本地 checkpoint、LoRA adapter、RAG 索引及 Runtime 依赖的环境中执行；仓库侧不能伪造该实验结果。
