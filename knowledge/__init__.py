"""knowledge/ —— 法律知识库 / RAG 数据层（原 legal_rag 的入库脚本 + inference/direct_rag.py）。

包含：
- ``ingest``：PDF -> 分块 -> 写入 Chroma 向量库（一次性脚本）。
- ``direct_rag``：绕过 LangChain、直接用 chromadb 原生客户端检索的轻量实现（预留/备用）。

MCP 协议相关的检索服务端在 ``mcp/server.py``，不在这里 —— knowledge/ 只负责"数据怎么
进向量库、怎么被直接查询"，mcp/ 负责"怎么把检索能力通过协议暴露给 Agent"。
"""
