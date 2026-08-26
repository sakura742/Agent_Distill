"""Long-running MCP transport for the legal tool service.

Tool metadata and domain-to-collection routing live in ``tools_config.json``
and ``ToolRegistry``. Retrieval implementation is isolated in
``LegalRetrieverService`` so the MCP transport stays thin and testable.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from app.logging_config import get_logger
from configs.settings import settings
from mcp_service.retriever_service import build_default_service
from mcp_service.tool_registry import ToolRegistry

logger = get_logger(__name__)
mcp_server = FastMCP("legal-assistant")
tool_registry = ToolRegistry(settings.tools_config_path)
retriever_service = build_default_service()


@mcp_server.tool()
def search_civil_law(query: str, limit: int = 3) -> str:
    """检索民法典相关法律条款。"""
    return retriever_service.search("search_civil_law", query, limit)


@mcp_server.tool()
def search_labor_law(query: str, limit: int = 3) -> str:
    """检索劳动法及劳动合同法相关法律条款。"""
    return retriever_service.search("search_labor_law", query, limit)


if __name__ == "__main__":
    logger.info("MCP legal-assistant server starting...")
    logger.info("registered tools: %s", [tool.name for tool in tool_registry.all()])
    mcp_server.run()
