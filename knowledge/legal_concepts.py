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

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1_000, "万": 10_000, "亿": 100_000_000}


def _chinese_number_to_int(text: str) -> int | None:
    """Convert common Chinese legal article numerals to Arabic integers."""
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
            continue
        unit = _CN_UNITS.get(char)
        if unit is None:
            return None
        if unit < 10_000:
            section += (number or 1) * unit
        else:
            section += number
            number = 0
            total += section * unit
            section = 0
    return total + section + number


def _normalize_article(article: str) -> str:
    value = str(article).strip().replace("第", "").replace("条", "")
    normalized = _chinese_number_to_int(value)
    return str(normalized) if normalized is not None else value


def article_concepts(domain: str, article: str | None) -> tuple[str, ...]:
    if domain != "civil" or not article:
        return ()
    return _CIVIL_ARTICLE_CONCEPTS.get(_normalize_article(article), ())
