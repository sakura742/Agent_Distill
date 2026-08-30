# -*- coding: utf-8 -*-
"""按法律领域构建独立 Chroma collection 的入库脚本。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.logging_config import get_logger
from configs.settings import settings
from knowledge.domain_config import LEGAL_CORPORA
from knowledge.legal_chunker import documents_from_pdf_pages

logger = get_logger(__name__)


def _build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_collection(
    corpus,
    *,
    reset: bool = False,
    embedding_model_name: str | None = None,
    chroma_db_dir: str | Path | None = None,
) -> int:
    file_path = settings.data_dir / corpus.filename
    if not file_path.exists():
        logger.warning("未找到法律文档: %s", file_path)
        return 0

    model_name = embedding_model_name or settings.embedding_model_name
    db_dir = Path(chroma_db_dir or settings.chroma_db_dir)
    logger.info(
        "开始解析 %s -> collection=%s, embedding=%s",
        corpus.filename, corpus.collection, model_name,
    )
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

    embeddings = _build_embeddings(model_name)
    db_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(db_dir))
            client.delete_collection(name=corpus.collection)
            logger.info("已删除旧 collection: %s", corpus.collection)
        except Exception:
            pass

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=corpus.collection,
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
            "embedding_normalized": True,
        },
        persist_directory=str(db_dir),
    )
    logger.info(
        "完成 %s: %d chunks (cosine, normalized embeddings, %s)",
        corpus.collection, len(docs), model_name,
    )
    return len(docs)


def build_all(
    *,
    reset: bool = False,
    embedding_model_name: str | None = None,
    chroma_db_dir: str | Path | None = None,
) -> dict[str, int]:
    return {
        corpus.domain: build_collection(
            corpus,
            reset=reset,
            embedding_model_name=embedding_model_name,
            chroma_db_dir=chroma_db_dir,
        )
        for corpus in LEGAL_CORPORA
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="构建分法域法律知识库")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--embedding-model", default=None, help="Embedding 模型名或本地路径")
    parser.add_argument("--chroma-dir", default=None, help="独立 Chroma 目录，用于 embedding A/B")
    args = parser.parse_args()
    logger.info(
        "知识库构建结果: %s",
        build_all(
            reset=args.reset,
            embedding_model_name=args.embedding_model,
            chroma_db_dir=args.chroma_dir,
        ),
    )


if __name__ == "__main__":
    main()
