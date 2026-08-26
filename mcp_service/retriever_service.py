# -*- coding: utf-8 -*-
"""MCP 与法律 RAG 之间的适配层。"""

from __future__ import annotations

from functools import lru_cache

from app.exceptions import KnowledgeBaseError
from knowledge.retriever import LegalRetriever
from mcp_service.tool_registry import ToolDefinition, ToolRegistry


class LegalRetrieverService:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    @lru_cache(maxsize=8)
    def _get_retriever(self, collection: str) -> LegalRetriever:
        for tool in self.registry.all():
            if tool.collection == collection:
                return LegalRetriever(tool.domain)
        raise KnowledgeBaseError(f"没有找到 collection 对应的法律领域: {collection}")

    def search(self, tool_name: str, query: str, limit: int = 3) -> str:
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        if limit < 1 or limit > 10:
            raise ValueError("limit 必须在 1~10 之间")
        tool: ToolDefinition = self.registry.get(tool_name)
        if not tool.collection:
            raise KnowledgeBaseError(f"工具 {tool_name} 未配置 collection")
        retriever = self._get_retriever(tool.collection)
        results = retriever.search(query, top_k=limit)
        if not results:
            return "未检索到相关法条。"
        return "\n\n".join(
            f"[{item.metadata.get('law_name', '未知法律')}"
            f" {item.metadata.get('article', '')}"
            f" | {item.metadata.get('source', '')}"
            f" p.{item.metadata.get('page', '')}]\n{item.content}"
            for item in results
        )


def build_default_service() -> LegalRetrieverService:
    from configs.settings import settings

    return LegalRetrieverService(ToolRegistry(settings.tools_config_path))
