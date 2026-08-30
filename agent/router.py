"""Hybrid legal-domain router with explicit non-legal abstention."""

from __future__ import annotations

import re
from dataclasses import dataclass

from configs.settings import settings

UNKNOWN_DOMAIN = "unknown"
SEMANTIC_MIN_SCORE = 0.42
SEMANTIC_MIN_MARGIN = 0.03
RULE_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class RouteResult:
    domain: str
    confidence: float
    candidates: list[dict[str, float | str]]
    method: str


_RULES = {
    "labor": (
        "劳动", "工资", "加班", "辞退", "解雇", "劳动合同", "社保", "工伤",
        "仲裁", "工作群", "降薪", "裁员", "经济补偿", "违法解除", "未签合同",
        "不签劳动合同", "续签", "加班费", "培训费", "培训期间", "服务期", "竞业", "年终奖",
        "试用期", "试用期间", "试用期被辞退", "辞职", "离职", "解除劳动关系",
    ),
    "civil": (
        "民法", "侵权", "借款", "借钱", "借条", "民间借贷", "债务人", "欠款",
        "合同", "违约", "买卖", "交货", "逾期交货", "租赁", "租房", "押金", "赔偿",
        "人格权", "婚姻", "邻居", "邻里", "楼上", "楼下", "漏水", "天花板",
        "房屋", "家具", "网购", "网购家具", "尺寸不符", "商品不符", "物业", "小区",
        "绿地", "车位", "业主", "财产损害", "相邻关系", "宠物", "交通事故", "人身损害",
    ),
}

_DOMAIN_DESCRIPTIONS = {
    "labor": "劳动关系 工资 加班 辞退 劳动合同 社保 工伤 劳动仲裁 降薪 裁员 经济补偿 培训费 竞业 试用期 辞职 离职",
    "civil": "民事法律关系 合同 侵权 借款 借钱 借条 民间借贷 债务人 欠款 买卖 逾期交货 租赁 押金 赔偿 商品买卖 网购 家具 尺寸不符 物业 小区 绿地 邻里漏水 房屋 财产损害 相邻关系 人身损害",
}


class HybridRouter:
    """Rules first; embeddings second; abstain when evidence is weak or ambiguous."""

    def __init__(self) -> None:
        self._embeddings = None

    def route(self, question: str) -> RouteResult:
        if not question or not question.strip():
            raise ValueError("question 不能为空")

        scores = self._rule_scores(question)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_score = ranked[0][1]
        second_score = ranked[1][1]

        if best_score > 0 and best_score > second_score:
            confidence = best_score / max(best_score + second_score, 1)
            if confidence >= RULE_MIN_CONFIDENCE:
                return RouteResult(
                    ranked[0][0], float(confidence),
                    [{"domain": d, "score": float(s)} for d, s in ranked], "rule",
                )

        semantic = self._semantic_scores(question)
        if semantic:
            ranked_semantic = sorted(semantic.items(), key=lambda x: x[1], reverse=True)
            best_domain, best_score = ranked_semantic[0]
            second_score = ranked_semantic[1][1] if len(ranked_semantic) > 1 else 0.0
            if best_score >= SEMANTIC_MIN_SCORE and best_score - second_score >= SEMANTIC_MIN_MARGIN:
                return RouteResult(
                    best_domain, float(best_score),
                    [{"domain": d, "score": float(s)} for d, s in ranked_semantic], "embedding",
                )
            return self._unknown_result(ranked_semantic, "embedding_abstain")

        return self._unknown_result(
            ranked if best_score > 0 else [],
            "rule_abstain" if best_score > 0 else "fallback",
        )

    @staticmethod
    def _unknown_result(candidates, method: str) -> RouteResult:
        return RouteResult(
            UNKNOWN_DOMAIN, 0.0,
            [{"domain": d, "score": float(s)} for d, s in candidates]
            or [{"domain": "labor", "score": 0.0}, {"domain": "civil", "score": 0.0}],
            method,
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

            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(y * y for y in b))
                return dot / (na * nb) if na and nb else 0.0

            return {domain: cosine(query, vector) for domain, vector in prototypes.items()}
        except Exception:
            return {}
