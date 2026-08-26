# -*- coding: utf-8 -*-
"""法律条款感知分块。"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from knowledge.legal_schema import LegalChunk

_ARTICLE_RE = re.compile(r"(?m)(?=第\s*[0-9一二三四五六七八九十百千万]+\s*条)")
_CHAPTER_RE = re.compile(r"(?m)^(第\s*[0-9一二三四五六七八九十百千万]+\s*[章节].*)$")
_LAW_TITLE_RE = re.compile(r"(?m)^\s*([^\n]{2,80}(?:法|条例|规定|办法|解释))\s*$")


def _clean(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _article_number(text: str) -> str | None:
    match = re.match(r"第\s*([0-9一二三四五六七八九十百千万]+)\s*条", text.strip())
    return match.group(1) if match else None


def _chapter_for(position: int, chapters: list[tuple[int, str]]) -> str | None:
    current = None
    for start, chapter in chapters:
        if start > position:
            break
        current = chapter
    return current


def split_legal_text(
    text: str,
    *,
    domain: str,
    law_name: str,
    source: str,
    page: int | None = None,
    max_chars: int = 1200,
) -> list[LegalChunk]:
    """优先按章节/条款切分，超长条款再做保留边界的二次切分。"""
    text = _clean(text)
    if not text:
        return []

    chapters = [(m.start(), m.group(1).strip()) for m in _CHAPTER_RE.finditer(text)]
    starts = [m.start() for m in _ARTICLE_RE.finditer(text)]
    parts: list[tuple[int, str]] = []
    if starts:
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            article_text = _clean(text[start:end])
            if article_text:
                parts.append((start, article_text))
    else:
        parts = [(0, text)]

    chunks: list[LegalChunk] = []
    for article_index, (position, article_text) in enumerate(parts):
        article = _article_number(article_text)
        chapter = _chapter_for(position, chapters)
        pieces = [article_text]
        if len(article_text) > max_chars:
            # 不再用原来的全局 500/50 滑窗；只对单个超长条款做局部分段。
            pieces = [article_text[i : i + max_chars] for i in range(0, len(article_text), max_chars)]
        for piece_index, piece in enumerate(pieces):
            chunk_id = f"{Path(source).stem}:{page or 0}:{article or article_index}:{piece_index}"
            chunks.append(
                LegalChunk(
                    text=piece,
                    domain=domain,
                    law_name=law_name,
                    source=source,
                    article=article,
                    chapter=chapter,
                    page=page,
                    chunk_id=chunk_id,
                )
            )
    return chunks


def documents_from_pdf_page(
    text: str,
    *,
    domain: str,
    law_name: str,
    source: str,
    page: int,
) -> list[Document]:
    chunks = split_legal_text(
        text,
        domain=domain,
        law_name=law_name,
        source=source,
        page=page,
    )
    return [Document(page_content=c.text, metadata=c.to_metadata()) for c in chunks]
