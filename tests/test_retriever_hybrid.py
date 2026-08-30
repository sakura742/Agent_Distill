from types import SimpleNamespace

from knowledge.retriever import LegalRetriever, RetrievedChunk, _lexical_overlap


def test_lexical_overlap_rewards_shared_chinese_phrases():
    high = _lexical_overlap("逾期交货", "买卖合同一方逾期交货，应承担违约责任")
    low = _lexical_overlap("逾期交货", "承租人未按约定支付租金")
    assert high > low


def test_hybrid_rerank_preserves_distance_and_adds_rerank_score(monkeypatch):
    retriever = LegalRetriever.__new__(LegalRetriever)
    retriever.domain = "civil"
    retriever.collection = "civil_law"
    retriever.vectorstore = SimpleNamespace(
        similarity_search_with_score=lambda *args, **kwargs: [
            (SimpleNamespace(page_content="承租人应当按照约定支付租金", metadata={"article": "七百二十二"}), 0.2),
            (SimpleNamespace(page_content="买卖合同逾期交货应承担违约责任", metadata={"article": "五百七十七"}), 0.3),
        ],
        _collection=SimpleNamespace(metadata={"hnsw:space": "l2"}, count=lambda: 2),
    )
    results = retriever.search("买卖合同逾期交货怎么办？", top_k=1, candidate_k=2, hybrid=True)
    assert len(results) == 1
    assert results[0].metadata["article"] == "五百七十七"
    assert results[0].distance == 0.3
    assert results[0].rerank_score is not None
