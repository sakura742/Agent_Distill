from types import SimpleNamespace

from mcp_service.retriever_service import LegalRetrieverService


def test_retrieval_service_filters_low_score_results(tmp_path):
    service = LegalRetrieverService.__new__(LegalRetrieverService)
    service.registry = None

    class FakeRetriever:
        def search(self, query, top_k=5, candidate_k=20, rerank=True):
            return [
                SimpleNamespace(content="高相关法条", metadata={"law_name": "劳动法", "article": "第四十四条", "source": "x.pdf", "page": 1}, score=0.82),
                SimpleNamespace(content="噪声法条", metadata={"law_name": "劳动法", "article": "第一百零三条", "source": "x.pdf", "page": 2}, score=0.31),
            ]

    service.collection_for_tool = lambda tool: "labor_law"
    service._get_retriever = lambda collection: FakeRetriever()

    result = service.search("search_labor_law", "加班费")
    assert "高相关法条" in result
    assert "噪声法条" not in result
    assert "score=0.820" in result


def test_retrieval_service_keeps_legacy_invoke_contract(tmp_path):
    service = LegalRetrieverService.__new__(LegalRetrieverService)
    service.registry = None

    class FakeRetriever:
        def invoke(self, query):
            return [SimpleNamespace(page_content=f"result:{query}")]

    service.collection_for_tool = lambda tool: "labor_law"
    service._get_retriever = lambda collection: FakeRetriever()
    assert service.search("search_labor_law", "劳动问题") == "result:劳动问题"
