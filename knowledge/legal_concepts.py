# -*- coding: utf-8 -*-
"""稳定的法律概念别名，用于增强法条索引，不参与答案生成。"""

from __future__ import annotations

# These are doctrine-level retrieval aliases, not benchmark answers.
# Keep them conservative and auditable; do not encode individual benchmark
# questions or gold references here.
_CIVIL_ARTICLE_CONCEPTS: dict[str, tuple[str, ...]] = {
    "1165": ("一般侵权责任", "过错责任", "侵权行为", "过错", "损害", "赔偿责任"),
    "1184": ("财产损失", "财产损害赔偿", "损失计算", "损失金额", "侵权赔偿"),
    "288": ("相邻关系", "不动产相邻关系", "相邻不动产权利人", "用水排水", "通风采光", "便利和损失"),
    "296": ("相邻关系", "不动产相邻", "自然流水", "用水排水", "相邻权益"),
    "1252": ("建筑物倒塌", "建筑物损害", "建筑物责任", "物件损害", "侵权责任"),
    "1253": ("建筑物脱落", "坠落物", "建筑物责任", "物件损害", "侵权责任"),
    "1254": ("高空抛物", "建筑物抛掷物", "抛掷物损害", "侵权责任"),
}


def article_concepts(domain: str, article: str | None) -> tuple[str, ...]:
    if domain != "civil" or not article:
        return ()
    return _CIVIL_ARTICLE_CONCEPTS.get(str(article), ())
