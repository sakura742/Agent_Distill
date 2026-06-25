# -*- coding: utf-8 -*-
import os
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM

# 沿用原 legal_rag 相同的持久化路径
CHROMA_DB_PATH = r"D:\py\Agent_Distill\legal_rag\chroma_db" 
COLLECTION_NAME = "legal_documents"

class DirectRetriever:
    def __init__(self):
        # 初始化本地 Chroma 客户端
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

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
            return f"本地检索发生异常: {e}"