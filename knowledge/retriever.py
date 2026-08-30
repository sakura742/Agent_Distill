# -*- coding: utf-8 -*-
"""法域隔离的 RAG Retriever，支持 Metadata Filter 与可选 CrossEncoder Rerank。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    distance: float | None = None
    rerank_score: float | None = None


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding checkpoint once per process."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


def _distance_to_score(distance: float, metric: str) -> float:
    """Convert Chroma distance to a monotonic [0,1] relevance score.

    Chroma commonly uses squared L2 distance by default.  It must not be
    converted with ``1 - distance / 2`` because L2 distances can exceed 2,
    which collapses many legitimate results to 0.  For cosine distance,
    ``1-distance`` is the natural similarity.  For L2 we use a bounded
    reciprocal transform; this is intentionally a ranking/diagnostic score,
    not a calibrated probability.
    """
    d = max(0.0, float(distance))
    metric = (metric or "l2").lower()
    if metric in {"cosine", "cos"}:
        return max(0.0, min(1.0, 1.0 - d))
    if metric in {"ip", "inner_product", "inner-product"}:
        return max(0.0, min(1.0, 1.0 - d))
    return 1.0 / (1.0 + d)


class LegalRetriever:
    def __init__(self, domain: str):
        corpus = corpus_by_domain(domain)
        self.domain = domain
        self.collection = corpus.collection
        self.embeddings = _get_embeddings()
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

    def _distance_metric(self) -> str:
        metadata = getattr(self.vectorstore._collection, "metadata", None) or {}
        return str(metadata.get("hnsw:space", "l2"))

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

        raw = self.vectorstore.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=filters,
        )
        metric = self._distance_metric()
        results = [
            RetrievedChunk(
                content=document.page_content,
                metadata=document.metadata,
                distance=float(distance),
                score=_distance_to_score(float(distance), metric),
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
            reranked = []
            for raw_score, item in zip(scores, results):
                reranked.append(
                    RetrievedChunk(
                        content=item.content,
                        metadata=item.metadata,
                        distance=item.distance,
                        score=item.score,
                        rerank_score=float(raw_score),
                    )
                )
            return [item for _, item in sorted(
                zip(scores, reranked), key=lambda pair: float(pair[0]), reverse=True
            )]
        except Exception:
            return results


def get_retriever(domain: str) -> LegalRetriever:
    return LegalRetriever(domain)
