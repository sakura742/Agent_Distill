# Phase 2：MCP Tool Service 施工记录

## 目标

把 Phase 1 中的 MCP 脚本升级为稳定的工具服务基础设施：

- `distill/tools_config.json` 作为工具元数据与参数 Schema 的单一真源；
- MCP Server 作为常驻进程运行，不在每次工具调用时重新创建服务进程；
- 工具契约显式记录 `domain` 与 `collection`，为后续法域隔离提供稳定接口；
- MCP 服务不允许在 collection 不存在时静默创建空 collection；
- MCP transport 与 RAG/retriever 实现解耦；
- 删除重复的 MCP 诊断脚本，并把可自动验证的逻辑迁移到 pytest。

## 本次改动

### 1. Tool Contract

`distill/tools_config.json` 从原来的：

```text
name / description / parameters
```

升级为：

```text
name / description / domain / collection / parameters
```

当前工具：

- `search_civil_law` → `civil` → `civil_law`
- `search_labor_law` → `labor` → `labor_law`

### 2. Tool Registry

新增 `mcp_service/tool_registry.py`。

职责：

- 读取工具配置；
- 校验配置结构；
- 检查重复工具名；
- 提供 `get(name)` / `all()` 查询接口；
- 将配置错误统一转换为 `ConfigurationError`。

Agent / MCP Server 后续只依赖 Registry，而不需要自己解析 JSON。

### 3. Retriever Service

新增 `mcp_service/retriever_service.py`，将向量库实现从 MCP transport 中抽离。

```text
MCP Server
   ↓
Tool Registry
   ↓
Tool Definition
   ↓
LegalRetrieverService
   ↓
collection
   ↓
Retriever
```

Retriever 按 collection 做 LRU Cache，常驻服务生命周期内复用已初始化的 retriever。

同时显式使用 Chroma PersistentClient 检查 collection 是否存在，避免 LangChain 在 collection
缺失时创建空 collection。

### 4. MCP Server

`mcp_service/server.py` 现在只负责 MCP transport 和工具暴露；实际检索由
`LegalRetrieverService` 执行。这样 Phase 4 接入 LangGraph 时，可以复用同一 Tool Contract，
而不需要让 Agent 了解 Chroma 的实现细节。

### 5. 法域路由验证

新增 `tests/test_mcp_tool_service.py`，在不依赖真实 Chroma 数据和 Embedding 模型的情况下验证：

- `search_civil_law` → `civil_law`；
- `search_labor_law` → `labor_law`；
- 实际 search 调用使用 Tool Contract 选定的 collection；
- 空 query 被拒绝。

这证明 Phase 2 的“工具 → 法域 → collection”运行时路由已经固定。真实 collection 的建立与
端到端检索仍属于 Phase 3。

### 6. 诊断脚本

删除 `mcp_service/test_mcp_server.py` 和 `mcp_service/debug_rag.py`，重复的诊断逻辑不再作为
业务模块存在。可自动验证的行为进入 `tests/`；真实 MCP stdio 全链路验证在 Phase 3 的知识库
建立后执行。

## 当前边界

Phase 2 不负责重新构建法律知识库。当前仓库的 Chroma 数据仍来自 Phase 1 的单 collection
入库方式，因此本阶段只建立“工具 → 法域 → collection”的契约和运行时基础设施。

真正创建 `civil_law` / `labor_law` 两个 collection，以及条款级 Metadata、分域入库和
检索评估，放在 Phase 3 完成。

在 Phase 3 完成之前，如果目标 collection 不存在，MCP 工具会明确抛出知识库错误，而不是
错误地从其他法域检索。

## 验收

Phase 2 的代码级验收范围：

- Tool Contract 单一配置源；
- Tool Registry 配置校验；
- MCP Server 常驻入口；
- Retriever Service 与 transport 解耦；
- collection 按工具契约隔离；
- 缺失 collection 显式报错；
- 单元测试覆盖核心路由逻辑。

由于本次修改通过 GitHub 远程编辑完成，**尚未声称已经在你的本地环境执行 pytest**。合并前
应在实际开发环境运行：

```bash
uv run pytest tests/
```

Phase 3 完成分域知识库后，再执行真实 MCP stdio 端到端检索验证。
