# -*- coding: utf-8 -*-
"""轻量检索器（原 inference/direct_rag.py，Phase 1 迁移至 knowledge/，因为它和
ingest.py 一样属于"直接操作向量库"的数据层代码，而不是 Agent 推理层）。

当前未被任何模块实际导入（与重构前一致，Phase 1 不新增调用方，只做搬迁 +
路径修复 + 统一异常）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

from configs.settings import settings
from app.exceptions import KnowledgeBaseError
from app.logging_config import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "legal_documents"


class DirectRetriever:
    def __init__(self):
        # 初始化本地 Chroma 客户端（路径来自 configs.settings，替代原硬编码的
        # r"D:\py\Agent_Distill\legal_rag\chroma_db"）
        self.client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise KnowledgeBaseError(
                f"无法打开 Chroma collection '{COLLECTION_NAME}'，"
                f"请先运行 knowledge/ingest.py 构建向量库。原始错误: {exc}"
            ) from exc

    def search(self, query: str, top_k: int = 3) -> str:
        """直接调用本地向量库检索，无 MCP 损耗"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            # 拼接检索到的文本片段
            documents = results.get("documents", [[]])[0]
            if not documents:
                return "未检索到相关法条。"
            return "\n\n".join(documents)
        except Exception as e:
            logger.error("本地检索发生异常: %s", e)
            return f"本地检索发生异常: {e}"
