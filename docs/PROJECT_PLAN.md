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
| 5 | Qwen3.5-2B + Trajectory 蒸馏 | ✅ | 2026-08-27 | phase5_distillation_notes.md |
| 6 | Benchmark / Evaluation 体系化 | 🔄 进行中 | 2026-08-27 | evaluation/README.md |
| 7 | 多轮对话 + Web Dashboard + Docker | ⬜ | — | — |

## Phase 6：Benchmark / Evaluation 体系化

### 目标

建立可重复、可量化、与 Runtime trace 对齐的 Agent 评测体系，覆盖：

`Routing → Retrieval → Tool Calling → Workflow → Answer → Multi-turn`

另行统计 Citation Precision / Recall。评测只使用可观测行为和实际 benchmark 数据，不虚构结果。

### 已实现

- `evaluation/metrics.py`：Accuracy、Recall@K、MRR、Citation Precision/Recall、F1、Answer token-overlap F1。
- `evaluation/benchmark.py`：统一 JSONL benchmark schema 和六类评测执行器。
- `evaluation/run_benchmark.py`：命令行评测入口。
- `evaluation/benchmark_schema.json`：benchmark 数据格式定义。
- `evaluation/README.md`：评测维度、字段和运行说明。

### 指标

| 能力 | 核心指标 |
|---|---|
| Routing | Routing Accuracy |
| Retrieval | Recall@5 / MRR |
| Tool Calling | Tool Selection Accuracy / Argument Accuracy |
| Workflow | Workflow Success Rate |
| Answer | Reference-aligned F1（当前提供 token-overlap baseline） |
| Citation | Citation Precision / Recall |
| Multi-turn | Conversation Success Rate |

### 运行

```bash
python -m evaluation.run_benchmark evaluation/benchmark.jsonl
```

### 当前边界

Phase 6 当前完成评测基础设施和指标定义，但**没有伪造 benchmark 数值**。下一步应准备 gold benchmark 数据，并分别对 Raw Qwen2.5、Phase 5 Qwen3.5、Teacher/Reference 跑同一套 benchmark，生成 baseline → distilled → teacher 的可比结果。

## 后续阶段

- **阶段 7**：Checkpoint、多轮会话、Web Dashboard、Docker 化。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-25 | Phase 0 / 0.5 / 1 完成 |
| 2026-08-26 | Phase 2 MCP Tool Service 完成 |
| 2026-08-27 | Phase 3 RAG 分域、条款感知 Chunk、Metadata Filter、Rerank 与检索指标完成 |
| 2026-08-27 | Phase 4 LangGraph Runtime、Hybrid Router、状态管理、条件工作流和 Re-plan 完成 |
| 2026-08-27 | Phase 5 Qwen3.5 Serving、Structured Trajectory、Hard Example Mining、4-bit LoRA SFT 完成 |
| 2026-08-27 | Phase 6 Benchmark / Evaluation 基础设施开工 |
