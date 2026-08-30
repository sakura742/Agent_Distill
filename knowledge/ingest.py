# -*- coding: utf-8 -*-
"""按法律领域构建独立 Chroma collection 的入库脚本。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # pymupdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.logging_config import get_logger
from configs.settings import settings
from knowledge.domain_config import LEGAL_CORPORA
from knowledge.legal_chunker import documents_from_pdf_pages

logger = get_logger(__name__)


def _build_embeddings() -> HuggingFaceEmbeddings:
    # Normalize document/query vectors so cosine distance has a stable meaning
    # and is bounded in the expected range. The same configuration is used by
    # knowledge.retriever at query time.
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_collection(corpus, *, reset: bool = False) -> int:
    file_path = settings.data_dir / corpus.filename
    if not file_path.exists():
        logger.warning("未找到法律文档: %s", file_path)
        return 0

    logger.info("开始解析 %s -> collection=%s", corpus.filename, corpus.collection)
    with fitz.open(file_path) as pdf_doc:
        pages = [page.get_text("text") for page in pdf_doc]

    docs = documents_from_pdf_pages(
        pages,
        domain=corpus.domain,
        law_name=corpus.law_name,
        source=corpus.filename,
    )
    if not docs:
        logger.warning("文档没有产生有效条款: %s", file_path)
        return 0

    embeddings = _build_embeddings()
    os.makedirs(settings.chroma_db_dir, exist_ok=True)

    if reset:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
            client.delete_collection(name=corpus.collection)
            logger.info("已删除旧 collection: %s", corpus.collection)
        except Exception:
            pass

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=corpus.collection,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(settings.chroma_db_dir),
    )
    logger.info("完成 %s: %d 个 chunks (cosine, normalized embeddings)", corpus.collection, len(docs))
    return len(docs)


def build_all(*, reset: bool = False) -> dict[str, int]:
    return {corpus.domain: build_collection(corpus, reset=reset) for corpus in LEGAL_CORPORA}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="构建分法域法律知识库")
    parser.add_argument("--reset", action="store_true", help="删除目标 collection 后重新入库")
    args = parser.parse_args()
    logger.info("知识库构建结果: %s", build_all(reset=args.reset))


if __name__ == "__main__":
    main()
