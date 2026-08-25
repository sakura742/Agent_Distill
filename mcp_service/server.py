"""MCP 工具服务端（原 legal_rag/server.py，Phase 1 迁移至 mcp_service/，
见 mcp_service/__init__.py 中关于为何不直接叫 mcp/ 的说明）。

工具注册、检索逻辑与"两个工具共用同一 retriever"的现状行为完全未变
（按要求 Phase 1 不实现真正的法域路由，那是后续阶段的 RAG 工作）。
仅将硬编码的 chroma_db 路径改为从 configs.settings 读取，并把裸
FileNotFoundError 换成统一的 KnowledgeBaseError。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from configs.settings import settings
from app.exceptions import KnowledgeBaseError
from app.logging_config import get_logger

logger = get_logger(__name__)

mcp_server = FastMCP("legal-assistant")

DB_DIR = settings.chroma_db_dir

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_top_k})
else:
    raise KnowledgeBaseError(
        f"向量库不存在: {DB_DIR}。请先运行 `python knowledge/ingest.py`！"
    )


# 【维持现状】统一工具名：拆分民法检索（两个工具函数体目前仍完全相同，
# 真正按法域路由属于 Phase 2 的 RAG 分域重构范围，Phase 1 不改动此行为）
@mcp_server.tool()
def search_civil_law(query: str, limit: int = 3) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs[:limit]])


@mcp_server.tool()
def search_labor_law(query: str, limit: int = 3) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs[:limit]])


if __name__ == "__main__":
    logger.info("MCP legal-assistant server starting...")
    mcp_server.run()
