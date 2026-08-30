# -*- coding: utf-8 -*-
"""法域隔离的 RAG Retriever，支持可配置 embedding、hybrid、query rewrite 和 CrossEncoder。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.exceptions import KnowledgeBaseError
from configs.settings import settings
from knowledge.domain_config import corpus_by_domain
from knowledge.query_rewrite import rewrite_legal_query


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    score: float
    distance: float | None = None
    rerank_score: float | None = None


@lru_cache(maxsize=8)
def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def _distance_to_score(distance: float, metric: str) -> float:
    d = max(0.0, float(distance))
    metric = (metric or "cosine").lower()
    if metric in {"cosine", "cos", "ip", "inner_product", "inner-product"}:
        return max(0.0, min(1.0, 1.0 - d))
    return 1.0 / (1.0 + d)


def _cn_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> set[str]:
    text = re.sub(r"\s+", "", text)
    if len(text) < n_min:
        return {text} if text else set()
    grams: set[str] = set()
    for n in range(n_min, min(n_max, len(text)) + 1):
        grams.update(text[i:i+n] for i in range(len(text) - n + 1))
    return grams


def _lexical_overlap(query: str, content: str) -> float:
    q, c = _cn_ngrams(query), _cn_ngrams(content)
    return len(q & c) / len(q) if q else 0.0


class LegalRetriever:
    def __init__(
        self,
        domain: str,
        *,
        embedding_model_name: str | None = None,
        chroma_db_dir: str | None = None,
        validate_index: bool = True,
    ):
        corpus = corpus_by_domain(domain)
        self.domain, self.collection = corpus.domain, corpus.collection
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name
        self.chroma_db_dir = str(chroma_db_dir or settings.chroma_db_dir)
        self.embeddings = _get_embeddings(self.embedding_model_name)
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_db_dir,
            )
            if self.vectorstore._collection.count() == 0:
                raise KnowledgeBaseError(f"法律知识库 collection '{self.collection}' 为空")
            if validate_index:
                self._validate_index_contract()
        except KnowledgeBaseError:
            raise
        except Exception as exc:
            raise KnowledgeBaseError(
                f"无法打开法律知识库 collection '{self.collection}'，请先运行 knowledge/ingest.py"
            ) from exc

    def _collection_metadata(self) -> dict[str, Any]:
        return dict(getattr(self.vectorstore._collection, "metadata", None) or {})

    def _validate_index_contract(self) -> None:
        metadata = self._collection_metadata()
        indexed_model = metadata.get("embedding_model")
        if indexed_model and str(indexed_model) != str(self.embedding_model_name):
            raise KnowledgeBaseError(
                "Embedding 模型与 Chroma 索引不一致: "
                f"index={indexed_model!r}, query={self.embedding_model_name!r}. "
                "请使用对应的 chroma_db_dir 或重新 ingest --reset。"
            )
        metric = str(metadata.get("hnsw:space", "cosine")).lower()
        if metric != "cosine":
            raise KnowledgeBaseError(
                f"Chroma collection '{self.collection}' 使用 {metric!r} metric；"
                "Phase 6 当前要求 cosine。请重新 ingest --reset。"
            )

        # Validate dimension early. This turns Chroma's low-level dimension error
        # into an actionable project-level error before the first query.
        try:
            expected_dim = len(self.embeddings.embed_query("法律检索维度检查"))
            stored = self.vectorstore._collection.get(limit=1, include=["embeddings"])
            vectors = stored.get("embeddings") if stored else None
            if vectors is not None and len(vectors) > 0 and vectors[0] is not None:
                actual_dim = len(vectors[0])
                if actual_dim != expected_dim:
                    raise KnowledgeBaseError(
                        "Embedding 维度与 Chroma 索引不一致: "
                        f"index={actual_dim}, query={expected_dim}. "
                        "请删除该 Chroma 目录后按同一 embedding 模型重新建库。"
                    )
        except KnowledgeBaseError:
            raise
        except Exception:
            # Some Chroma versions don't expose embeddings via get(); metadata
            # and the query contract still provide the primary guard.
            pass

    def _distance_metric(self) -> str:
        return str(self._collection_metadata().get("hnsw:space", "cosine"))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        law_name: str | None = None,
        article: str | None = None,
        rerank: bool = False,
        hybrid: bool = False,
        rewrite: bool = False,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")

        search_query = rewrite_legal_query(query, self.domain) if rewrite else query
        candidate_k = max(candidate_k or top_k * settings.retrieval_candidate_multiplier, top_k)
        filters: dict[str, Any] = {"domain": self.domain}
        if law_name:
            filters["law_name"] = law_name
        if article:
            filters["article"] = article

        raw = self.vectorstore.similarity_search_with_score(search_query, k=candidate_k, filter=filters)
        metric = self._distance_metric()
        results = [
            RetrievedChunk(
                document.page_content,
                document.metadata,
                _distance_to_score(float(distance), metric),
                float(distance),
            )
            for document, distance in raw
        ]

        if hybrid and len(results) > 1:
            semantic_max = max(item.score for item in results)
            semantic_min = min(item.score for item in results)
            span = semantic_max - semantic_min
            ranked: list[RetrievedChunk] = []
            for item in results:
                semantic_rank = (item.score - semantic_min) / span if span > 1e-12 else 0.0
                lexical = _lexical_overlap(search_query, item.content)
                ranked.append(RetrievedChunk(
                    item.content,
                    {**item.metadata, "lexical_score": lexical},
                    item.score,
                    item.distance,
                    0.70 * semantic_rank + 0.30 * lexical,
                ))
            results = sorted(ranked, key=lambda item: float(item.rerank_score or 0.0), reverse=True)

        if rerank and len(results) > 1:
            results = self._rerank(search_query, results)
        return results[:top_k]

    @staticmethod
    def _rerank(query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        model_name = getattr(settings, "reranker_model_name", None)
        if not model_name:
            raise KnowledgeBaseError(
                "已请求 rerank，但 AGENT_DISTILL_RERANKER_MODEL 未配置。"
                "请配置可用的 CrossEncoder 模型后再运行 rerank 实验。"
            )
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name)
            scores = model.predict([(query, item.content) for item in results])
            reranked = [RetrievedChunk(
                item.content, item.metadata, item.score, item.distance, float(raw_score)
            ) for raw_score, item in zip(scores, results)]
            return sorted(reranked, key=lambda item: float(item.rerank_score or 0.0), reverse=True)
        except Exception as exc:
            raise KnowledgeBaseError(
                f"CrossEncoder reranker 加载或推理失败: {model_name!r}"
            ) from exc


def get_retriever(
    domain: str,
    *,
    embedding_model_name: str | None = None,
    chroma_db_dir: str | None = None,
) -> LegalRetriever:
    return LegalRetriever(domain, embedding_model_name=embedding_model_name, chroma_db_dir=chroma_db_dir)
