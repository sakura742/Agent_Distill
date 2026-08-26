# Agent_Distill → Legal Agent Platform 项目计划

> 本文档持续维护，记录“现在做到哪一步了”，是团队/协作者了解项目当前状态的唯一入口。
> 每个阶段完成或状态变化时更新本文档。

## 状态总览

| 阶段 | 名称 | 状态 | 完成日期 | 详细记录 |
|---|---|---|---|---|
| 0 | 代码审计 | ✅ 完成 | 2026-08-25 | architecture/current_architecture.md |
| 0.5 | 目标态规划 | ✅ 完成 | 2026-08-25 | architecture/refactor_plan.md |
| 1 | 模块化骨架 + 配置/日志/异常系统 | ✅ 完成 | 2026-08-25 | phase1_refactor_notes.md |
| 2 | MCP Tool Service 常驻化 + 工具契约统一 | ✅ 完成 | 2026-08-26 | phase2_mcp_notes.md |
| 3 | RAG 分域重构（Metadata / Rerank） | 🔄 进行中 | 2026-08-26 | phase3_rag_notes.md |
| 4 | LangGraph Agent Runtime + Hybrid Router | ⬜ 未开始 | — | — |
| 5 | 模型层升级（Qwen3.5-2B）+ Trajectory 蒸馏 | ⬜ 未开始 | — | — |
| 6 | Benchmark / Evaluation 体系化 | ⬜ 未开始 | — | — |
| 7 | 多轮对话 + Web Dashboard + Docker 化 | ⬜ 未开始 | — | — |

## 阶段 3：RAG 分域重构（进行中）

**目标**：替换原来的全库 500/50 定长滑窗和单 collection 架构，使法律条款成为可追踪、可过滤、可评估的检索单元。

### 已实现

- 新增 `knowledge/domain_config.py`：定义 civil / labor 两个法律语料域及 collection 映射。
- 新增 `knowledge/legal_schema.py`：统一 `LegalChunk` 数据结构。
- 新增 `knowledge/legal_chunker.py`：优先按章节、法律条款切分；超长单条款才进行局部分段；保留 `law_name/chapter/article/source/page/chunk_id` 元数据。
- 重构 `knowledge/ingest.py`：按 PDF 文件分别写入 `civil_law` / `labor_law` Chroma collection，并保留页码信息。
- 新增 `knowledge/retriever.py`：按法域打开独立 collection，支持 `domain/law_name/article` Metadata Filter、候选召回和可选 CrossEncoder Rerank。
- `configs/settings.py` 增加 `AGENT_DISTILL_RERANKER_MODEL` 配置；未配置时不启用 CrossEncoder，不影响基础检索。
- MCP Retriever Service 改为调用统一 `knowledge.retriever.LegalRetriever`，从 Tool Contract 的 collection 路由到对应法域知识库。
- 新增 `evaluation/retrieval_metrics.py`：实现 Recall@K、MRR，为 Phase 6 Benchmark 提供可复用指标实现。
- 新增 chunker / retrieval metrics 单元测试。

### 数据构建

在本地准备 `data/minfa.pdf` 与 `data/labor_law.pdf` 后运行：

```bash
uv run python knowledge/ingest.py --reset
```

目标为两个独立 collection：`civil_law`、`labor_law`。

### 当前边界

代码已完成，但没有在远程环境执行本地 PDF 入库和 GPU/Embedding 推理，因此不能虚构 collection 数量、Recall@K、MRR 或检索延迟。需要在你的本地环境完成真实入库和 Benchmark 后才能将 Phase 3 标记为完成。

## 后续阶段概要

- **阶段 4**：LangGraph Runtime + Hybrid Router；Intent → Router → Agent → Planner → Tool → Retrieval → Generation → Verification。
- **阶段 5**：Qwen3.5-2B Serving + Agent Trajectory 蒸馏 + LoRA / Hard Example Mining。
- **阶段 6**：Routing / Retrieval / Tool Calling / Workflow / Answer / Multi-turn 六类 Benchmark。
- **阶段 7**：Checkpoint、多轮对话、Web Dashboard、Docker 化。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-25 | Phase 0 / 0.5 / 1 完成 |
| 2026-08-26 | Phase 2 MCP Tool Service 完成 |
| 2026-08-26 | Phase 3 开工：领域语料配置、条款感知 Chunk、分 collection 入库、Metadata Filter、可选 Rerank、Recall@K / MRR |
