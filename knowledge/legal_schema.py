# -*- coding: utf-8 -*-
"""领域知识库的统一数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LegalChunk:
    """一个可被向量检索的法律条款片段。"""

    text: str
    domain: str
    law_name: str
    source: str
    article: str | None = None
    chapter: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "domain": self.domain,
            "law_name": self.law_name,
            "source": self.source,
        }
        if self.article:
            result["article"] = self.article
        if self.chapter:
            result["chapter"] = self.chapter
        if self.page is not None:
            result["page"] = self.page
        if self.chunk_id:
            result["chunk_id"] = self.chunk_id
        result.update(self.metadata)
        return result
