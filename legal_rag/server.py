import os
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

mcp_server = FastMCP("legal-assistant")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    vectorstore = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    raise FileNotFoundError("Database not found. Please run `python ingest.py` first!")

# 【修复 BUG-1】统一工具名：拆分民法检索
@mcp_server.tool()
def search_civil_law(query: str, limit: int = 3) -> str:
    docs = retriever.invoke(query) 
    return "\n\n".join([doc.page_content for doc in docs[:limit]])

# 【修复 BUG-1】统一工具名：拆分劳动法检索
@mcp_server.tool()
def search_labor_law(query: str, limit: int = 3) -> str:
    docs = retriever.invoke(query) 
    return "\n\n".join([doc.page_content for doc in docs[:limit]])

if __name__ == "__main__":
    mcp_server.run()