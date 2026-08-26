# Agent_Distill → Legal Agent Platform 项目计划

> 本文持续维护，记录项目当前状态，是团队/协作者了解项目进度的入口。

## 状态总览

| 阶段 | 名称 | 状态 | 完成日期 | 详细记录 |
|---|---|---|---|---|
| 0 | 代码审计 | ✅ 完成 | 2026-08-25 | architecture/current_architecture.md |
| 0.5 | 目标态规划 | ✅ 完成 | 2026-08-25 | architecture/refactor_plan.md |
| 1 | 模块化骨架 + 配置/日志/异常系统 | ✅ 完成 | 2026-08-25 | phase1_refactor_notes.md |
| 2 | MCP Tool Service 常驻化 + 工具契约统一 | ✅ 完成 | 2026-08-26 | phase2_mcp_notes.md |
| 3 | RAG 分域重构（Metadata / Rerank） | ✅ 完成 | 2026-08-27 | phase3_rag_notes.md |
| 4 | LangGraph Agent Runtime + Hybrid Router | 🔄 进行中 | 2026-08-27 | phase4_agent_runtime_notes.md |
| 5 | 模型层升级（Qwen3.5-2B）+ Trajectory 蒸馏 | ⬜ 未开始 | — | — |
| 6 | Benchmark / Evaluation 体系化 | ⬜ 未开始 | — | — |
| 7 | 多轮对话 + Web Dashboard + Docker 化 | ⬜ 未开始 | — | — |

## 阶段 4：LangGraph Agent Runtime + Hybrid Router（进行中）

### 目标

把 Phase 2 的 Tool Service 和 Phase 3 的 Legal RAG 编排成状态化 Agent Runtime：

```text
Intent Analysis → Task Planning → Tool Decision → Tool Execution
→ Retrieval → Generation → Verification
                              ↑                 │
                              └──── Re-plan ───┘
```

### 已实现

- `agent/runtime/state.py`：统一 AgentState，保存问题、领域、意图、计划、工具调用、检索结果、引用、答案、验证状态、错误和执行 trace。
- `agent/router.py`：Hybrid Router，当前采用规则优先、Embedding 相似度补充、fallback 兜底；支持 labor / civil 两个法域。
- `agent/runtime/nodes.py`：实现 Intent Analysis、Task Planning、Tool Decision、Tool Execution、Retrieval、Generation、Verification 和 Re-plan 节点。
- `agent/runtime/graph.py`：使用 LangGraph StateGraph 构建条件工作流；验证失败或工具异常时进入一次 Re-plan。
- `tests/test_phase4_runtime.py`：通过 Fake Tool Service 验证路由、状态流转、引用生成和 trace。
- Generation 使用可注入 `answer_generator`，暂不把模型加载逻辑耦合到 Graph；Phase 5 接入 Qwen3.5-2B Serving。

### 当前边界

Phase 4 已完成 Runtime 骨架和可测试的工作流编排，但尚未标记完成：真实 MCP Transport、真实本地模型 Generation、多轮 Checkpoint 和 Benchmark 属于后续集成范围。

## 后续阶段概要

- **阶段 5**：Qwen3.5-2B Serving + Agent Trajectory 蒸馏 + LoRA / Hard Example Mining。
- **阶段 6**：Routing / Retrieval / Tool Calling / Workflow / Answer / Multi-turn 六类 Benchmark。
- **阶段 7**：Checkpoint、多轮对话、Web Dashboard、Docker 化。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-25 | Phase 0 / 0.5 / 1 完成 |
| 2026-08-26 | Phase 2 MCP Tool Service 完成 |
| 2026-08-27 | Phase 3 RAG 分域、条款感知 Chunk、Metadata Filter、Rerank 与检索指标完成并经本地验证 |
| 2026-08-27 | Phase 4 开工：LangGraph Runtime、Hybrid Router、状态管理、条件工作流和 Re-plan |
