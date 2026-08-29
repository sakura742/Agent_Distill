# -*- coding: utf-8 -*-
"""法域隔离的 RAG Retriever，支持 Metadata Filter 与可选 CrossEncoder Rerank。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.exceptions import KnowledgeBaseError
from configs.settings import settings
from knowledge.domain_config import corpus_by_domain


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    score: float


class LegalRetriever:
    def __init__(self, domain: str):
        corpus = corpus_by_domain(domain)
        self.domain = domain
        self.collection = corpus.collection
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection,
                embedding_function=self.embeddings,
                persist_directory=str(settings.chroma_db_dir),
            )
            if self.vectorstore._collection.count() == 0:
                raise KnowledgeBaseError(f"法律知识库 collection '{self.collection}' 为空")
        except KnowledgeBaseError:
            raise
        except Exception as exc:
            raise KnowledgeBaseError(
                f"无法打开法律知识库 collection '{self.collection}'，请先运行 knowledge/ingest.py"
            ) from exc

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        law_name: str | None = None,
        article: str | None = None,
        rerank: bool = False,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")

        candidate_k = max(candidate_k or top_k * 4, top_k)
        filters: dict[str, Any] = {"domain": self.domain}
        if law_name:
            filters["law_name"] = law_name
        if article:
            filters["article"] = article

        # Chroma's `similarity_search_with_relevance_scores` applies a
        # relevance function whose range assumptions vary across Chroma
        # versions/collection distance settings.  The benchmark only needs a
        # stable ranking, so read the native distance and convert cosine
        # distance to a bounded [0, 1] similarity ourselves.
        raw = self.vectorstore.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=filters,
        )
        results = [
            RetrievedChunk(
                content=document.page_content,
                metadata=document.metadata,
                score=max(0.0, min(1.0, 1.0 - float(distance) / 2.0)),
            )
            for document, distance in raw
        ]

        if rerank and len(results) > 1:
            results = self._rerank(query, results)
        return results[:top_k]

    @staticmethod
    def _rerank(query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """使用 CrossEncoder 做二阶段排序；模型不可用时保留向量排序结果。"""
        model_name = getattr(settings, "reranker_model_name", None)
        if not model_name:
            return results
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name)
            pairs = [(query, item.content) for item in results]
            scores = model.predict(pairs)
            return [
                item for _, item in sorted(
                    zip(scores, results), key=lambda pair: float(pair[0]), reverse=True
                )
            ]
        except Exception:
            # Reranker 属于可选增强，不应让基础 RAG 因模型缺失而不可用。
            return results


def get_retriever(domain: str) -> LegalRetriever:
    return LegalRetriever(domain)
