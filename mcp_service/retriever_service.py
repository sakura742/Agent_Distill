"""Legal retrieval service used by MCP tools.

This module keeps vector-store concerns out of the MCP transport layer. Tool
names, domains and collection names come from ToolRegistry; the service only
resolves the configured collection and executes retrieval.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.exceptions import KnowledgeBaseError
from configs.settings import settings
from mcp_service.tool_registry import ToolRegistry


class LegalRetrieverService:
    """Long-lived retrieval service with per-collection retriever caching."""

    def __init__(self, registry: ToolRegistry, db_dir: Path, embedding_model_name: str, top_k: int = 3) -> None:
        self.registry = registry
        self.db_dir = Path(db_dir)
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    @lru_cache(maxsize=8)
    def _get_retriever(self, collection: str):
        """Resolve an existing Chroma collection and cache its retriever."""
        if not self.db_dir.exists() or not any(self.db_dir.iterdir()):
            raise KnowledgeBaseError(
                f"向量库不存在: {self.db_dir}。请先运行知识库入库流程。"
            )

        try:
            client = chromadb.PersistentClient(path=str(self.db_dir))
            client.get_collection(collection)
            vectorstore = Chroma(
                collection_name=collection,
                persist_directory=str(self.db_dir),
                embedding_function=self.embeddings,
            )
            return vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        except Exception as exc:
            raise KnowledgeBaseError(
                f"无法加载法域 collection '{collection}'，请确认对应知识库已完成入库。"
            ) from exc

    def search(self, tool_name: str, query: str, limit: int = 3) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")

        definition = self.registry.get(tool_name)
        safe_limit = max(1, min(int(limit), 10))
        retriever = self._get_retriever(definition.collection)
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs[:safe_limit])

    def collection_for_tool(self, tool_name: str) -> str:
        return self.registry.get(tool_name).collection


def build_default_service() -> LegalRetrieverService:
    return LegalRetrieverService(
        registry=ToolRegistry(settings.tools_config_path),
        db_dir=settings.chroma_db_dir,
        embedding_model_name=settings.embedding_model_name,
        top_k=settings.retrieval_top_k,
    )
