# Agent_Distill → Legal Agent Platform 项目计划

> 本文档持续维护，记录“现在做到哪一步了”，是团队/协作者了解项目当前状态的唯一入口。
> 每个阶段完成或状态变化时更新本文档。远期目标态设计见
> [`architecture/refactor_plan.md`](architecture/refactor_plan.md)；每个阶段的详细
> 施工记录单独成文（如 [`phase1_refactor_notes.md`](phase1_refactor_notes.md)）。

## 状态总览

| 阶段 | 名称 | 状态 | 完成日期 | 详细记录 |
|---|---|---|---|---|
| 0 | 代码审计 | ✅ 完成 | 2026-08-25 | [`architecture/current_architecture.md`](architecture/current_architecture.md) |
| 0.5 | 目标态规划 | ✅ 完成 | 2026-08-25 | [`architecture/refactor_plan.md`](architecture/refactor_plan.md) |
| 1 | 模块化骨架 + 配置/日志/异常系统 | ✅ 完成 | 2026-08-25 | [`phase1_refactor_notes.md`](phase1_refactor_notes.md) |
| 2 | MCP Tool Service 常驻化 + 工具契约统一 | 🔄 进行中 | — | 本分支施工中 |
| 3 | RAG 分域重构（Metadata / Rerank） | ⬜ 未开始 | — | — |
| 4 | LangGraph Agent Runtime + Hybrid Router | ⬜ 未开始 | — | — |
| 5 | 模型层升级（Qwen3.5-2B）+ Trajectory 蒸馏 | ⬜ 未开始 | — | — |
| 6 | Benchmark / Evaluation 体系化 | ⬜ 未开始 | — | — |
| 7 | 多轮对话 + Web Dashboard + Docker 化 | ⬜ 未开始 | — | — |

图例：✅ 完成　🔄 进行中　⬜ 未开始　⏸ 暂停

---

## 阶段 1：模块化骨架（已完成，2026-08-25）

**目标**：在不改变任何业务逻辑（训练核心、数据集格式、推理算法）的前提下，把项目
拆成 `app/agent/knowledge/mcp_service/distill/evaluation/web/deployment/tests/configs/docs`
十个模块，建立配置/日志/异常三个基础设施。

**验收结果**：`pytest tests/` → `26 passed, 12 skipped`（skip 均为沙盒缺三方库，
非路径错误），0 failed / 0 error。详见 phase1_refactor_notes.md §四。

**遗留给后续阶段、本阶段明确不处理的问题**（原样保留，未修复）：

- `mcp_service/server.py` 里 `search_civil_law` / `search_labor_law` 两个工具函数体
  仍然完全相同（共用同一 retriever，未做真正的法域路由）→ 阶段 2/3 处理。
- 训练数据 `output` 字段仍是自然语言拼接的伪 JSON，且约 1/6 样本存在 `keyword`/`query`
  参数名混用 → 阶段 5（Agent Trajectory 蒸馏）处理。
- `evaluation/evaluate.py` 测试集仍是硬编码在代码里的 5 道题，只评估“工具选对没有”
  一个维度 → 阶段 6（Benchmark 体系化）处理。
- `agent/inference_core.py` 每次请求串行加载/卸载两个模型、`web/app.py` 会话状态是
  进程内内存字典 → 阶段 4/7 处理。
- `knowledge/direct_rag.py` 仍是未被调用的预留代码。`mcp_service/debug_rag.py` 与
  `mcp_service/test_mcp_server.py` 原本功能重叠，Phase 2 已合并为保留 `debug_rag.py`
  一个诊断入口。

---

## 阶段 2：MCP Tool Service（进行中）

**目标**：把 MCP 从“脚本级工具”升级为稳定的工具服务基础设施，让上层 LangGraph
只依赖统一 Tool Contract，而不直接依赖 Chroma 或具体法律知识库实现。

### 已完成

- `distill/tools_config.json` 增加 `domain`、`collection` 字段，并统一参数 Schema。
- 新增 `mcp_service/tool_registry.py`，负责加载、校验并提供只读 Tool Registry。
- `mcp_service/server.py` 改为基于 Tool Registry 查找 collection，按工具法域获取并缓存
  Retriever。
- 禁止 collection 不存在时静默创建空 collection，避免法律法域错误被伪装成“无检索结果”。
- 删除与 `debug_rag.py` 功能高度重叠的 `test_mcp_server.py`。
- 新增 `tests/test_tool_registry.py`，覆盖正常加载、重复工具名和缺失配置等情况。

### 当前边界

Phase 2 只建立工具服务和 collection 路由契约；真正的 `civil_law` / `labor_law`
collection 由 Phase 3 的 RAG 分域入库创建。这样可以避免在 RAG Metadata 尚未重构前，
为了让 MCP“看起来能跑”而引入错误的法域回退。

### 待验收

- 完成 Phase 3 分域 collection 后，运行 MCP 端到端检索验证。
- 确认 stdio MCP Client → Server → Domain Retriever 全链路正常。
- 确认工具 Schema 与后续 LangGraph Tool Calling 输入格式一致。

---

## 阶段 3～7：概要（详见 refactor_plan.md）

以下为后续阶段简述，正式启动某阶段时会在本文档补充独立小节（目标、验收标准、实际改动、遗留问题）。

- **阶段 3 · RAG 分域重构**：条款感知分块替代定长字符切分；结构化 Chunk Metadata
  （law_name/chapter/article）；建立 `civil_law` / `labor_law` collection；Metadata
  前置过滤 + Rerank；Recall@K / MRR 可衡量。
- **阶段 4 · LangGraph Runtime + Hybrid Router**：`Intent→Router→Agent→Planner→Tool→
  Retrieval→Generation→Verification` 显式 Graph；失败自动 Re-plan；规则+语义+LLM 三级
  路由；按法律专业领域拆分多个 `BaseLegalAgent`。
- **阶段 5 · 模型层 + Trajectory 蒸馏**：接入 Qwen3.5-2B；生产 Serving（vLLM/TGI）
  替代现装现卸；训练数据升级为结构化 Agent Trajectory；五级数据质量流水线；
  DPO 偏好对齐；困难样本挖掘闭环。
- **阶段 6 · Benchmark 体系化**：独立可扩展测试集（建议 500–1000+ 条，分维度配比）；
  Routing / Retrieval / Tool Calling / Workflow / Answer / Multi-turn 六类指标；
  Base / LoRA / Agent Distilled 三方对比；train/val/test 隔离。
- **阶段 7 · 多轮对话 + Dashboard + Docker**：会话状态迁移 Redis/Postgres；结构化
  长期槽位记忆；Web Dashboard 拆分 Chat / Agent Trace / RAG Trace / Multi-turn State /
  Model Compare / Evaluation Dashboard / Error Analysis 七个页面；`deployment/` 补齐
  Dockerfile + docker-compose。

---

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-25 | 创建本文档；记录阶段 0 / 0.5 / 1 已完成 |
| 2026-08-25 | 阶段 1 收尾修复：`distill/train.py` stdout 副作用 bug、根 README 路径更新、补齐 `docs/phase1_refactor_notes.md`、`pyproject.toml` 补 `pytest`/`python-dotenv` |
| 2026-08-25 | 阶段 2 开工：统一 Tool Contract、增加 Tool Registry、MCP collection 路由基础设施、删除重复诊断脚本、增加 Registry 单元测试 |
