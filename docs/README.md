# docs/

- `architecture/current_architecture.md` — Phase 1 重构之前，对仓库当时状态
  (`legal_rag/` `distill/` `inference/` `web/` ...) 的只读代码审计报告。
  **保留原文不变**，作为"重构前快照"存档；文中提到的路径在 Phase 1 之后已不存在，
  这是预期的，不代表文档过期。
- `architecture/refactor_plan.md` — 基于上述审计写的目标态规划（LangGraph
  Runtime / Hybrid Router / 分域 RAG / Benchmark 体系等），描述的是 Phase 1
  之后多个阶段的远期目标，**Phase 1 本身不实现其中任何一项**（无 LangGraph、
  无真正的 MCP 分域路由、无新 RAG 能力），仅完成该计划里"阶段 0"的模块化骨架
  搭建部分。
- `phase1_refactor_notes.md` — Phase 1 实际做了什么、为什么、如何验证，
  是本次重构的权威记录。
- `legacy/` — 原 `version/` 目录整体迁移至此（`Agent_Distill_ProjectDoc.md`、
  `improvement_plan.md`、`improvement_plan_v2.md`），项目最早期的设计文档归档，
  内容未改动。
