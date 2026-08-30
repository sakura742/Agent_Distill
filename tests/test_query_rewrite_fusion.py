from types import SimpleNamespace

from knowledge.retriever import LegalRetriever


def _chunk(article: str, score: float):
    return SimpleNamespace(
        content=f"内容 {article}",
        metadata={"chunk_id": article, "law_name": "民法典", "article": article},
        score=score,
        distance=1.0 - score,
        rerank_score=None,
    )


def test_rewrite_fusion_preserves_original_hit():
    original = [_chunk("五百七十七", 0.90), _chunk("六百七十五", 0.70)]
    rewritten = [_chunk("六百七十五", 0.95), _chunk("六百七十三", 0.80)]
    merged = LegalRetriever._merge_query_variants(original, rewritten)
    articles = [item.metadata["article"] for item in merged]
    assert "五百七十七" in articles
    assert "六百七十五" in articles
    assert "六百七十三" in articles
    assert merged[0].metadata["article"] == "六百七十五"
