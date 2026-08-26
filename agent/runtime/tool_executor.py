"""Tool execution adapters for the LangGraph runtime.

The runtime depends on this small contract instead of knowing whether a tool
runs in-process or through MCP. Phase 4 tests use DirectToolExecutor; the
MCP adapter is available for integration runs.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from configs.settings import settings


class DirectToolExecutor:
    def __init__(self, service=None):
        self.service = service

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self.service is None:
            from mcp_service.retriever_service import build_default_service
            self.service = build_default_service()
        return self.service.search(
            tool_name,
            arguments["query"],
            int(arguments.get("limit", 5)),
        )


class MCPToolExecutor:
    """Execute one tool call through the Phase 2 FastMCP server."""

    def __init__(self, server_path=None):
        self.server_path = str(server_path or settings.mcp_server_path)

    async def _execute_async(self, tool_name: str, arguments: dict[str, Any]) -> str:
        params = StdioServerParameters(
            command="python",
            args=["-u", self.server_path],
            env=os.environ.copy(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    timeout=15.0,
                )
                texts = [item.text for item in response.content if hasattr(item, "text")]
                return "\n".join(texts) if texts else "未检索到相关法条。"

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        return asyncio.run(self._execute_async(tool_name, arguments))
