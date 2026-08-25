"""常驻 MCP Tool Service。

Phase 2 将工具元数据、参数 Schema、法域和 collection 名称统一交给
``distill/tools_config.json`` 管理；服务启动时只初始化 MCP Server 和工具注册表。
具体向量库 collection 的构建属于 Phase 3，当前若 collection 尚未建立，工具会返回
明确的知识库错误，而不会静默回退到错误法域。
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from mcp.server.fastmcp import FastMCP
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.exceptions import KnowledgeBaseError
from app.logging_config import get_logger
from configs.settings import settings
from mcp_service.tool_registry import ToolRegistry

logger = get_logger(__name__)
mcp_server = FastMCP("legal-assistant")
tool_registry = ToolRegistry(settings.tools_config_path)
embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


@lru_cache(maxsize=8)
def _get_retriever(collection: str):
    """按工具契约指定的 collection 创建并缓存 retriever。"""
    db_dir = settings.chroma_db_dir
    if not db_dir.exists() or not any(db_dir.iterdir()):
        raise KnowledgeBaseError(
            f"向量库不存在: {db_dir}。请先运行知识库入库流程。"
        )

    try:
        # 先显式检查 collection，避免 LangChain Chroma 在 collection 不存在时
        # 静默创建空 collection，进而把法域错误伪装成“检索无结果”。
        client = chromadb.PersistentClient(path=str(db_dir))
        client.get_collection(collection)
        vectorstore = Chroma(
            collection_name=collection,
            persist_directory=str(db_dir),
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_top_k})
    except Exception as exc:
        raise KnowledgeBaseError(
            f"无法加载法域 collection '{collection}'，请确认对应知识库已完成入库。"
        ) from exc


def _search(tool_name: str, query: str, limit: int = 3) -> str:
    if not query or not query.strip():
        raise ValueError("query 不能为空")
    definition = tool_registry.get(tool_name)
    limit = max(1, min(int(limit), 10))
    retriever = _get_retriever(definition.collection)
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs[:limit])


@mcp_server.tool()
def search_civil_law(query: str, limit: int = 3) -> str:
    """检索民法典相关法律条款。"""
    return _search("search_civil_law", query, limit)


@mcp_server.tool()
def search_labor_law(query: str, limit: int = 3) -> str:
    """检索劳动法及劳动合同法相关法律条款。"""
    return _search("search_labor_law", query, limit)


if __name__ == "__main__":
    logger.info("MCP legal-assistant server starting...")
    logger.info("registered tools: %s", [tool.name for tool in tool_registry.all()])
    mcp_server.run()
