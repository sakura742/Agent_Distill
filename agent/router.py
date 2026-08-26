"""Hybrid legal-domain router: rules first, embeddings second, fallback last."""

from __future__ import annotations

import re
from dataclasses import dataclass

from configs.settings import settings


@dataclass(frozen=True)
class RouteResult:
    domain: str
    confidence: float
    candidates: list[dict[str, float | str]]
    method: str


_RULES = {
    "labor": ("劳动", "工资", "加班", "辞退", "解雇", "劳动合同", "社保", "工伤", "仲裁", "工作群"),
    "civil": ("民法", "侵权", "借款", "合同", "违约", "买卖", "租赁", "赔偿", "人格权", "婚姻"),
}

_DOMAIN_DESCRIPTIONS = {
    "labor": "劳动关系 工资 加班 辞退 劳动合同 社保 工伤 劳动仲裁",
    "civil": "民事法律关系 合同 侵权 借款 买卖 租赁 赔偿 人格权 婚姻",
}


class HybridRouter:
    """Prefer deterministic rules; use embedding similarity when rules are inconclusive."""

    def __init__(self) -> None:
        self._embeddings = None

    def route(self, question: str) -> RouteResult:
        if not question or not question.strip():
            raise ValueError("question 不能为空")

        scores = self._rule_scores(question)
        if max(scores.values()) > 0:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            total = sum(scores.values()) or 1
            confidence = ranked[0][1] / total
            return RouteResult(
                domain=ranked[0][0],
                confidence=float(confidence),
                candidates=[{"domain": d, "score": float(s)} for d, s in ranked],
                method="rule",
            )

        semantic = self._semantic_scores(question)
        if semantic:
            ranked = sorted(semantic.items(), key=lambda x: x[1], reverse=True)
            return RouteResult(
                domain=ranked[0][0],
                confidence=float(ranked[0][1]),
                candidates=[{"domain": d, "score": float(s)} for d, s in ranked],
                method="embedding",
            )

        return RouteResult(
            domain="civil",
            confidence=0.0,
            candidates=[{"domain": "civil", "score": 0.0}, {"domain": "labor", "score": 0.0}],
            method="fallback",
        )

    @staticmethod
    def _rule_scores(question: str) -> dict[str, int]:
        text = re.sub(r"\s+", "", question)
        return {
            domain: sum(text.count(keyword) for keyword in keywords)
            for domain, keywords in _RULES.items()
        }

    def _semantic_scores(self, question: str) -> dict[str, float]:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            if self._embeddings is None:
                self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
            query = self._embeddings.embed_query(question)
            prototypes = {
                domain: self._embeddings.embed_query(description)
                for domain, description in _DOMAIN_DESCRIPTIONS.items()
            }
            import math

            def cosine(a: list[float], b: list[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(y * y for y in b))
                return dot / (na * nb) if na and nb else 0.0

            raw = {domain: cosine(query, vector) for domain, vector in prototypes.items()}
            low = min(raw.values())
            high = max(raw.values())
            if high <= 0:
                return {}
            return {domain: (score - low) / (high - low) if high != low else 1.0 for domain, score in raw.items()}
        except Exception:
            return {}
