# -*- coding: utf-8 -*-
"""MCP 与法律 RAG 之间的适配层。"""

from __future__ import annotations

from functools import lru_cache

from app.exceptions import KnowledgeBaseError
from knowledge.retriever import LegalRetriever
from mcp_service.tool_registry import ToolDefinition, ToolRegistry


DEFAULT_MIN_RELEVANCE = 0.45
DEFAULT_CANDIDATE_MULTIPLIER = 4


class LegalRetrieverService:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def collection_for_tool(self, tool_name: str) -> str:
        tool: ToolDefinition = self.registry.get(tool_name)
        if not tool.collection:
            raise KnowledgeBaseError(f"工具 {tool_name} 未配置 collection")
        return tool.collection

    @lru_cache(maxsize=8)
    def _get_retriever(self, collection: str) -> LegalRetriever:
        for tool in self.registry.all():
            if tool.collection == collection:
                return LegalRetriever(tool.domain)
        raise KnowledgeBaseError(f"没有找到 collection 对应的法律领域: {collection}")

    @staticmethod
    def _search_retriever(retriever, query: str, limit: int):
        search = getattr(retriever, "search", None)
        if callable(search):
            return search(
                query,
                top_k=limit,
                candidate_k=max(limit * DEFAULT_CANDIDATE_MULTIPLIER, limit),
                rerank=True,
                hybrid=True,
            )
        invoke = getattr(retriever, "invoke", None)
        if callable(invoke):
            return invoke(query)
        raise AttributeError("retriever 必须提供 search(query, top_k=...) 或 invoke(query)")

    def search(self, tool_name: str, query: str, limit: int = 3) -> str:
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        if limit < 1 or limit > 10:
            raise ValueError("limit 必须在 1~10 之间")

        collection = self.collection_for_tool(tool_name)
        retriever = self._get_retriever(collection)
        results = self._search_retriever(retriever, query, limit)
        if not results:
            return "未检索到相关法条。"

        def content_of(item):
            return getattr(item, "content", getattr(item, "page_content", ""))

        def metadata_of(item):
            return getattr(item, "metadata", {}) or {}

        def score_of(item):
            try:
                return float(getattr(item, "score", 0.0))
            except (TypeError, ValueError):
                return 0.0

        scored = [item for item in results if hasattr(item, "score")]
        if scored:
            filtered = [item for item in scored if score_of(item) >= DEFAULT_MIN_RELEVANCE]
            results = filtered if filtered else scored[:1]

        if all(not metadata_of(item) for item in results):
            return "\n\n".join(content_of(item) for item in results)

        return "\n\n".join(
            f"[{metadata_of(item).get('law_name', '未知法律')}"
            f" {metadata_of(item).get('article', '')}"
            f" | {metadata_of(item).get('source', '')}"
            f" p.{metadata_of(item).get('page', '')}"
            f" | score={score_of(item):.3f}"
            f" | rerank_score={getattr(item, 'rerank_score', None)!s}]\n{content_of(item)}"
            for item in results
        )


def build_default_service() -> LegalRetrieverService:
    from configs.settings import settings
    return LegalRetrieverService(ToolRegistry(settings.tools_config_path))
