# -*- coding: utf-8 -*-
"""法律文档与向量 collection 的法域配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LegalCorpus:
    domain: str
    collection: str
    law_name: str
    filename: str


LEGAL_CORPORA = (
    LegalCorpus("civil", "civil_law", "中华人民共和国民法典", "minfa.pdf"),
    LegalCorpus("labor", "labor_law", "中华人民共和国劳动法及相关法律", "labor_law.pdf"),
)


def corpus_by_domain(domain: str) -> LegalCorpus:
    for corpus in LEGAL_CORPORA:
        if corpus.domain == domain:
            return corpus
    raise KeyError(f"未知法律领域: {domain}")


def corpus_by_filename(filename: str) -> LegalCorpus:
    name = Path(filename).name
    for corpus in LEGAL_CORPORA:
        if corpus.filename == name:
            return corpus
    raise KeyError(f"未配置法律文档: {name}")
