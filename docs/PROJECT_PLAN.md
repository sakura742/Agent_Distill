# Agent_Distill → Legal Agent Platform 项目计划

> 本文档持续维护，记录“现在做到哪一步了”的项目当前状态，是团队/协作者了解项目进度的唯一入口。

## 状态总览

| 阶段 | 名称 | 状态 | 完成日期 | 详细记录 |
|---|---|---|---|---|
| 0 | 代码审计 | ✅ 完成 | 2026-08-25 | architecture/current_architecture.md |
| 0.5 | 目标态规划 | ✅ 完成 | 2026-08-25 | architecture/refactor_plan.md |
| 1 | 模块化骨架 + 配置/日志/异常系统 | ✅ 完成 | 2026-08-25 | phase1_refactor_notes.md |
| 2 | MCP Tool Service 常驻化 + 工具契约统一 | ✅ 完成 | 2026-08-26 | phase2_mcp_notes.md |
| 3 | RAG 分域重构（Metadata / Rerank） | ✅ 完成 | 2026-08-27 | phase3_rag_notes.md |
| 4 | LangGraph Agent Runtime + Hybrid Router | ✅ 完成 | 2026-08-27 | phase4_agent_runtime_notes.md |
| 5 | 模型层升级（Qwen3.5-2B）+ Trajectory 蒸馏 | ✅ 完成 | 2026-08-27 | phase5_distillation_notes.md |
| 6 | Benchmark / Evaluation 体系化 | ⬜ 未开始 | — | — |
| 7 | 多轮对话 + Web Dashboard + Docker 化 | ⬜ 未开始 | — | — |

## 阶段 4：LangGraph Agent Runtime + Hybrid Router

Phase 4 已完成 Runtime 骨架：Intent Analysis、Task Planning、Tool Decision、Tool Execution、Retrieval、Generation、Verification、Re-plan，以及规则优先 + Embedding 的 Hybrid Router。Generation 保持 `answer_generator` 注入点，因此 Phase 5 可以独立接入本地模型。

## 阶段 5：Qwen3.5-2B Serving + Agent Trajectory 蒸馏

### 目标

```text
Teacher → Structured Trajectory → Hard Example Mining → Qwen3.5-2B LoRA
                                                        ↓
Phase 4 Runtime ← Qwen3.5 Local Serving ← LoRA Adapter
```

### 已实现

- `configs/settings.py`：新增 Qwen3.5 模型、LoRA、trajectory、hard-example 路径及本地模型配置。
- `model_service/qwen35.py`：Qwen3.5-2B 常驻模型服务，复用模型实例，不再每个请求重复加载。
- `model_service/server.py`：FastAPI `/health` 和 `/v1/chat/completions`。
- `agent/runtime/qwen35_generator.py`：把 Serving 适配到 Phase 4 Generation Node。
- `distill/trajectory.py`：通过 Phase 4 Graph 生成结构化 Agent trajectory，并由教师模型生成最终答案。
- `distill/hard_mining.py`：基于 verification、retry、route confidence、tool/citation 缺失筛选 hard examples。
- `distill/train_phase5.py`：4-bit NF4 + LoRA + Gradient Checkpointing 的 Qwen3.5-2B SFT。
- `tests/test_phase5_distillation.py`：覆盖配置隔离、困难样本规则和 trajectory 序列化。
- `docs/phase5_distillation_notes.md`：完整记录 Phase 5 数据流、训练配置和运行方式。

### 训练目标

不直接训练隐藏 CoT，而是蒸馏可观测 Agent 行为：

`domain → intent → plan → tool → evidence → answer`

### 当前边界

Phase 5 完成模型 Serving、轨迹蒸馏、Hard Example Mining 和 LoRA 训练基础设施。Benchmark 数值对比进入 Phase 6；Checkpoint、多轮会话和 Web/Docker 进入 Phase 7。

## 后续阶段概要

- **阶段 6**：Routing / Retrieval / Tool Calling / Workflow / Answer / Multi-turn 六类 Benchmark。
- **阶段 7**：Checkpoint、多轮对话、Web Dashboard、Docker 化。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-25 | Phase 0 / 0.5 / 1 完成 |
| 2026-08-26 | Phase 2 MCP Tool Service 完成 |
| 2026-08-27 | Phase 3 RAG 分域、条款感知 Chunk、Metadata Filter、Rerank 与检索指标完成 |
| 2026-08-27 | Phase 4 LangGraph Runtime、Hybrid Router、状态管理、条件工作流和 Re-plan 完成 |
| 2026-08-27 | Phase 5 Qwen3.5 Serving、Structured Trajectory、Hard Example Mining、4-bit LoRA SFT 完成 |
