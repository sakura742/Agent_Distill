"""Hybrid legal-domain router with explicit non-legal abstention."""

from __future__ import annotations

import re
from dataclasses import dataclass

from configs.settings import settings

UNKNOWN_DOMAIN = "unknown"
SEMANTIC_MIN_SCORE = 0.42
SEMANTIC_MIN_MARGIN = 0.03

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
    """Rules first; embeddings second; abstain when semantic evidence is weak."""
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
            return RouteResult(ranked[0][0], float(confidence), [{"domain": d, "score": float(s)} for d, s in ranked], "rule")
        semantic = self._semantic_scores(question)
        if semantic:
            ranked = sorted(semantic.items(), key=lambda x: x[1], reverse=True)
            best_domain, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            if best_score >= SEMANTIC_MIN_SCORE and best_score - second_score >= SEMANTIC_MIN_MARGIN:
                return RouteResult(best_domain, float(best_score), [{"domain": d, "score": float(s)} for d, s in ranked], "embedding")
            return self._unknown_result(ranked, "embedding_abstain")
        return self._unknown_result([], "fallback")

    @staticmethod
    def _unknown_result(candidates, method: str) -> RouteResult:
        return RouteResult(UNKNOWN_DOMAIN, 0.0, [{"domain": d, "score": float(s)} for d, s in candidates] or [{"domain": "labor", "score": 0.0}, {"domain": "civil", "score": 0.0}], method)

    @staticmethod
    def _rule_scores(question: str) -> dict[str, int]:
        text = re.sub(r"\s+", "", question)
        return {domain: sum(text.count(keyword) for keyword in keywords) for domain, keywords in _RULES.items()}

    def _semantic_scores(self, question: str) -> dict[str, float]:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            if self._embeddings is None:
                self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
            query = self._embeddings.embed_query(question)
            prototypes = {domain: self._embeddings.embed_query(description) for domain, description in _DOMAIN_DESCRIPTIONS.items()}
            import math
            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
                return dot / (na * nb) if na and nb else 0.0
            return {domain: cosine(query, vector) for domain, vector in prototypes.items()}
        except Exception:
            return {}
