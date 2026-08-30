# -*- coding: utf-8 -*-
"""将法律章节与条款概念转换为稳定的检索主题词。"""
from __future__ import annotations

import re

from knowledge.legal_concepts import article_concepts


_TOPIC_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("侵权责任",), "一般侵权 过错责任 损害赔偿 财产损害 人身损害 相邻关系"),
    (("合同编", "合同"), "合同履行 违约责任 迟延履行 合同纠纷 损害赔偿"),
    (("物权",), "物权保护 不动产 相邻关系 占有 使用 收益"),
    (("人格权",), "人格权 人身权益 损害赔偿"),
    (("婚姻家庭", "婚姻"), "婚姻家庭 夫妻 财产 抚养"),
    (("劳动",), "劳动关系 工资 劳动合同 解除 补偿 仲裁"),
)


def chapter_topics(chapter: str | None) -> str:
    if not chapter:
        return ""
    text = re.sub(r"\s+", "", chapter)
    topics: list[str] = []
    for triggers, expansion in _TOPIC_RULES:
        if any(trigger in text for trigger in triggers):
            topics.append(expansion)
    return " ".join(dict.fromkeys(topics))


def enriched_retrieval_text(
    *,
    law_name: str,
    article: str | None,
    chapter: str | None,
    text: str,
) -> str:
    """Build an embedding-oriented text without changing the legal wording."""
    parts = [law_name]
    if chapter:
        parts.append(chapter)
        topics = chapter_topics(chapter)
        if topics:
            parts.append(f"法律主题：{topics}")
    domain = "civil" if "民法" in law_name else "labor" if "劳动" in law_name else ""
    concepts = article_concepts(domain, article)
    if concepts:
        parts.append("法律概念：" + " ".join(concepts))
    if article:
        parts.append(f"第{article}条")
    parts.append(text)
    return "\n".join(parts)
