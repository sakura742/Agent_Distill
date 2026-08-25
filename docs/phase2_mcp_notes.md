# Phase 2：MCP Tool Service 施工记录

## 目标

把 Phase 1 中的 MCP 脚本升级为稳定的工具服务基础设施：

- `distill/tools_config.json` 作为工具元数据与参数 Schema 的单一真源；
- MCP Server 常驻运行，不在每次工具调用时重新创建服务进程；
- 工具契约显式记录 `domain` 与 `collection`，为后续法域隔离提供稳定接口；
- MCP 服务不允许在 collection 不存在时静默创建空 collection；
- 删除重复的 MCP 诊断脚本。

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

### 3. MCP Server

`mcp_service/server.py` 改为：

```text
MCP Server
   ↓
Tool Registry
   ↓
Tool Definition
   ↓
collection
   ↓
Retriever
```

Retriever 使用 LRU Cache 缓存，避免同一法域在常驻服务生命周期内反复初始化。

同时显式使用 Chroma PersistentClient 检查 collection 是否存在，避免 LangChain 在 collection
缺失时创建空 collection。

### 4. 诊断脚本

删除 `mcp_service/test_mcp_server.py`，保留 `debug_rag.py` 作为 MCP stdio 全链路诊断入口。

## 当前边界

Phase 2 不负责重新构建法律知识库。当前仓库的 Chroma 数据仍来自 Phase 1 的单 collection
入库方式，因此本阶段只建立“工具 → 法域 → collection”的契约和运行时基础设施。

真正创建 `civil_law` / `labor_law` 两个 collection，以及条款级 Metadata、分域入库和
检索评估，放在 Phase 3 完成。

在 Phase 3 完成之前，如果目标 collection 不存在，MCP 工具会明确抛出知识库错误，而不是
错误地从其他法域检索。

## 测试

新增 `tests/test_tool_registry.py`，覆盖：

- 正常工具配置加载；
- 重复工具名拒绝；
- 配置文件不存在时拒绝。

由于当前修改通过 GitHub 远程编辑完成，本次施工记录不宣称已经在本地环境执行 pytest；
合并前应在实际开发环境运行：

```bash
uv run pytest tests/
```

并在 Phase 3 完成后执行 MCP stdio 集成测试。
