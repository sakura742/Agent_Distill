"""mcp_service/ —— MCP（Model Context Protocol）工具服务与诊断脚本。

命名说明（与用户给定的模块清单 `mcp/` 的唯一偏差，其余全部按清单原样建立）：
项目依赖官方 `mcp` PyPI 包（pyproject.toml 中 `mcp>=1.27.2`），
`server.py`/`agent_pipeline.py`/`debug_rag.py`/`test_mcp_server.py` 都要
`from mcp.server.fastmcp import FastMCP` / `from mcp import ClientSession` 等
导入这个第三方包。如果本地目录也叫 `mcp/` 且带 `__init__.py`，只要项目根目录在
`sys.path` 里（几乎所有运行方式都会这样），Python 会优先解析到本地目录，
彻底屏蔽掉真正的第三方 `mcp` 包，导致所有 MCP 相关功能在导入阶段直接报错。
这违反了"不删除现有功能""保持项目可运行"的硬性要求，所以 Phase 1 把这一个目录
改名为 `mcp_service/`，避免和第三方包重名；其余 8 个目录（app/agent/knowledge/
distill/evaluation/web/deployment/tests/configs/docs）均使用用户指定的原名。

包含：
- ``server``：FastMCP 服务端，暴露 ``search_civil_law`` / ``search_labor_law`` 两个工具。
- ``debug_rag`` / ``test_mcp_server``：两个 MCP 客户端连通性诊断脚本（功能高度重叠，
  Phase 1 按"不删除现有功能"原则两者都保留原样迁移，去重留给后续阶段）。

Phase 1 未改变 MCP 协议交互逻辑，只做搬迁 + import 修复。
"""
