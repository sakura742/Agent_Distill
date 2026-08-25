"""知识库入库脚本（原 legal_rag/ingest.py，Phase 1 迁移至 knowledge/）。

功能与分块算法（500 字 / 50 重叠的滑窗切分）完全未变，仅将原先硬编码的
"BASE_DIR/../data"、"legal_rag/chroma_db" 路径改为从 configs.settings 读取，
并将 print 换成统一 logger。
"""

import os
import sys

# 允许 `python knowledge/ingest.py` 直接运行（脚本方式）时也能找到项目根目录下的
# configs/ 和 app/ 包，与原项目 web/app.py 里 sys.path.insert 的写法保持一致。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # pymupdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from configs.settings import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        chunks.append(text[start:start + size])
        start += (size - overlap)
    return chunks


def main():
    data_dir = settings.data_dir
    db_dir = settings.chroma_db_dir

    logger.info("PDF 检索目录: %s", data_dir)
    logger.info("向量库保存目录: %s", db_dir)

    logger.info("开始读取 PDF 文件...")
    pdf_files = ["labor_law.pdf", "minfa.pdf"]
    docs = []

    for file_name in pdf_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            # 使用 pymupdf 替代原 PyPDFLoader
            pdf_doc = fitz.open(file_path)
            full_text = ""
            for page in pdf_doc:
                full_text += page.get_text("text") + "\n"

            # 进行自定义文本分块
            chunks = chunk_text(full_text, size=500, overlap=50)

            # 封装为兼容对象
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": file_name, "chunk_id": i}
                ))
            logger.info("成功解析并分块: %s (共 %d 块)", file_name, len(chunks))
            pdf_doc.close()
        else:
            logger.warning("未找到文件: %s", file_path)

    if not docs:
        logger.error("无有效文件，退出执行。")
        return

    logger.info("正在加载 Embedding 模型 (%s)...", settings.embedding_model_name)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    logger.info("正在写入 Chroma 向量库...")
    os.makedirs(db_dir, exist_ok=True)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(db_dir),
    )
    logger.info("向量库构建完成！")


if __name__ == "__main__":
    main()
