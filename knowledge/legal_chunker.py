# -*- coding: utf-8 -*-
"""法律条款感知分块，并为检索索引加入可审计的法律概念别名。"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from knowledge.legal_concepts import article_concepts
from knowledge.legal_schema import LegalChunk
from knowledge.topic_enrichment import enriched_retrieval_text

_ARTICLE_RE = re.compile(r"(?m)(?=^\s*第\s*[0-9一二三四五六七八九十百千万亿零〇两]+\s*条\b)")
_CHAPTER_RE = re.compile(r"(?m)^\s*(第\s*[0-9一二三四五六七八九十百千万亿零〇两]+\s*[章节].*)$")
_ARTICLE_HEADER_ONLY_RE = re.compile(r"^\s*第\s*[0-9一二三四五六七八九十百千万亿零〇两]+\s*条\s*$")


def _clean(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _article_number(text: str) -> str | None:
    match = re.match(r"第\s*([0-9一二三四五六七八九十百千万亿零〇两]+)\s*条", text.strip())
    return match.group(1) if match else None


def _is_article_header_only(text: str) -> bool:
    return bool(_ARTICLE_HEADER_ONLY_RE.fullmatch(text.strip()))


def _chapter_for(position: int, chapters: list[tuple[int, str]]) -> str | None:
    current = None
    for start, chapter in chapters:
        if start > position:
            break
        current = chapter
    return current


def split_legal_text(text: str, *, domain: str, law_name: str, source: str, page: int | None = None, max_chars: int = 1200) -> list[LegalChunk]:
    """Split a complete corpus text by article boundaries."""
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
            if article_text and not _is_article_header_only(article_text):
                parts.append((start, article_text))
    else:
        parts = [(0, text)]

    chunks: list[LegalChunk] = []
    for article_index, (position, article_text) in enumerate(parts):
        article = _article_number(article_text)
        chapter = _chapter_for(position, chapters)
        pieces = [article_text]
        if len(article_text) > max_chars:
            pieces = [article_text[i : i + max_chars] for i in range(0, len(article_text), max_chars)]
        for piece_index, piece in enumerate(pieces):
            chunk_id = f"{Path(source).stem}:{page or 0}:{article or article_index}:{piece_index}"
            chunks.append(LegalChunk(text=piece, domain=domain, law_name=law_name, source=source, article=article, chapter=chapter, page=page, chunk_id=chunk_id))
    return chunks


def split_legal_text_with_pages(pages: list[str], *, domain: str, law_name: str, source: str, max_chars: int = 1200) -> list[LegalChunk]:
    """Split an entire PDF at article boundaries while preserving start page."""
    cleaned_pages = [_clean(page) for page in pages]
    combined = "\n\n".join(page for page in cleaned_pages if page)
    if not combined:
        return []

    page_offsets: list[tuple[int, int]] = []
    cursor = 0
    for page in cleaned_pages:
        start = cursor
        end = start + len(page)
        page_offsets.append((start, end))
        cursor = end + 2

    def page_for(offset: int) -> int | None:
        for page_number, (start, end) in enumerate(page_offsets, 1):
            if start <= offset <= end:
                return page_number
        return None

    chapters = [(m.start(), m.group(1).strip()) for m in _CHAPTER_RE.finditer(combined)]
    starts = [m.start() for m in _ARTICLE_RE.finditer(combined)]
    if not starts:
        return split_legal_text(combined, domain=domain, law_name=law_name, source=source, page=1, max_chars=max_chars)

    chunks: list[LegalChunk] = []
    for article_index, start in enumerate(starts):
        end = starts[article_index + 1] if article_index + 1 < len(starts) else len(combined)
        article_text = _clean(combined[start:end])
        if not article_text or _is_article_header_only(article_text):
            continue
        article = _article_number(article_text)
        chapter = _chapter_for(start, chapters)
        start_page = page_for(start)
        pieces = [article_text]
        if len(article_text) > max_chars:
            pieces = [article_text[i : i + max_chars] for i in range(0, len(article_text), max_chars)]
        for piece_index, piece in enumerate(pieces):
            chunk_id = f"{Path(source).stem}:{start_page or 0}:{article or article_index}:{piece_index}"
            chunks.append(LegalChunk(text=piece, domain=domain, law_name=law_name, source=source, article=article, chapter=chapter, page=start_page, chunk_id=chunk_id))
    return chunks


def _document_for_chunk(chunk: LegalChunk) -> Document:
    retrieval_text = enriched_retrieval_text(law_name=chunk.law_name, article=chunk.article, chapter=chunk.chapter, text=chunk.text)
    concepts = article_concepts(chunk.domain, chunk.article)
    if concepts:
        retrieval_text += "\n法律概念：" + " ".join(concepts)
    metadata = chunk.to_metadata()
    if concepts:
        metadata["legal_concepts"] = list(concepts)
    metadata["original_text"] = chunk.text
    metadata["retrieval_enriched"] = bool(concepts) or bool(chunk.chapter)
    return Document(page_content=retrieval_text, metadata=metadata)


def documents_from_pdf_page(text: str, *, domain: str, law_name: str, source: str, page: int) -> list[Document]:
    chunks = split_legal_text(text, domain=domain, law_name=law_name, source=source, page=page)
    return [_document_for_chunk(c) for c in chunks]


def documents_from_pdf_pages(pages: list[str], *, domain: str, law_name: str, source: str) -> list[Document]:
    chunks = split_legal_text_with_pages(pages, domain=domain, law_name=law_name, source=source)
    return [_document_for_chunk(c) for c in chunks]
