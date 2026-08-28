# Agent_Distill → Legal Agent Platform 项目计划

> 持续维护的项目状态入口。

## 状态总览

| 阶段 | 名称 | 状态 | 完成日期 | 详细记录 |
|---|---|---|---|---|
| 0 | 代码审计 | ✅ | 2026-08-25 | architecture/current_architecture.md |
| 0.5 | 目标态规划 | ✅ | 2026-08-25 | architecture/refactor_plan.md |
| 1 | 模块化骨架 + 配置/日志/异常 | ✅ | 2026-08-25 | phase1_refactor_notes.md |
| 2 | MCP Tool Service + 工具契约 | ✅ | 2026-08-26 | phase2_mcp_notes.md |
| 3 | RAG 分域 + Metadata / Rerank | ✅ | 2026-08-27 | phase3_rag_notes.md |
| 4 | LangGraph Runtime + Hybrid Router | ✅ | 2026-08-27 | phase4_agent_runtime_notes.md |
| 5 | Qwen3.5-4B + Trajectory 蒸馏 | ✅ | 2026-08-27 | phase5_distillation_notes.md |
| 6 | Benchmark / Evaluation 体系化 | 🔄 进行中 | 2026-08-28 | evaluation/README.md |
| 7 | 多轮对话 + Web Dashboard + Docker | ⬜ | — | — |

## Phase 6：Benchmark / Evaluation 体系化

### 主实验

仅比较同一模型的两个 checkpoint 状态：

`Qwen3.5-4B Raw` vs `Qwen3.5-4B LoRA`

不纳入 Qwen2.5、Qwen3.5-2B 或 Teacher 作为主对照组。

两组必须共享同一 benchmark、RAG corpus、Tool Contract、Runtime、generation 配置和 evaluation code，以隔离 LoRA 蒸馏带来的增益。

### 评测维度

`Routing → Retrieval → Tool Calling → Workflow → Answer → Citation → Multi-turn`

核心指标：Routing Accuracy、Recall@5、MRR、Tool Selection/Argument Accuracy、Workflow Success Rate、Answer F1、Citation Precision/Recall、Multi-turn Success Rate，以及平均推理延迟。

### 已实现

- `evaluation/metrics.py`：指标实现。
- `evaluation/benchmark.py`：统一 benchmark schema / evaluator。
- `evaluation/run_benchmark.py`：离线 benchmark CLI。
- `evaluation/run_runtime_benchmark.py`：Runtime 路由基线检查。
- `evaluation/experiment.py`：Raw / LoRA paired experiment。
- `evaluation/compare_models.py`：统一对照实验 CLI。
- `evaluation/run_qwen35_benchmark.py`：真实 Qwen3.5-4B Raw / LoRA Runtime benchmark。
- `evaluation/benchmark_schema.json`：数据格式。
- `evaluation/experiment_config.json`：实验控制变量。
- `evaluation/README.md`：评测说明。
- `data/evaluation/benchmark.jsonl`：当前 benchmark 样本。

### 运行

```bash
python -m evaluation.run_qwen35_benchmark
```

输出：

```text
data/evaluation/results/qwen35_4b_raw.json
data/evaluation/results/qwen35_4b_lora.json
```

### 当前状态

评测基础设施和真实模型 Runner 已完成；最终数值必须在本地 Qwen3.5-4B Raw 与 Phase 5 LoRA checkpoint 可用后运行生成，不预填实验结果。

## 后续阶段

- **阶段 7**：Checkpoint、多轮会话、Web Dashboard、Docker 化。
