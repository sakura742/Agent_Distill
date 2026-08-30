# -*- coding: utf-8 -*-
"""Deterministic legal query normalization/expansion for retrieval experiments."""

from __future__ import annotations

from typing import Iterable


_CIVIL_EXPANSIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("漏水", "天花板", "渗水"), "侵权责任 相邻关系 财产损害 赔偿"),
    (("借钱", "借款", "借条", "欠款", "不还", "拒绝还款"), "借款合同 返还借款 到期还款 债务人 民间借贷"),
    (("逾期交货", "迟延交货", "不交货"), "买卖合同 迟延履行 违约责任 损害赔偿"),
    (("押金", "房东不退", "租房"), "租赁合同 押金返还 承租人 出租人 违约责任"),
    (("交通事故", "撞车", "被撞"), "侵权责任 人身损害 财产损害 损害赔偿"),
)


def rewrite_legal_query(query: str, domain: str) -> str:
    """Expand common lay terms into stable legal concepts without changing intent."""
    text = " ".join(str(query).strip().split())
    if domain != "civil" or not text:
        return text
    additions: list[str] = []
    for triggers, expansion in _CIVIL_EXPANSIONS:
        if any(trigger in text for trigger in triggers):
            additions.append(expansion)
    if not additions:
        return text
    return text + " " + " ".join(dict.fromkeys(additions))


def rewrite_batch(queries: Iterable[str], domain: str) -> list[str]:
    return [rewrite_legal_query(q, domain) for q in queries]
