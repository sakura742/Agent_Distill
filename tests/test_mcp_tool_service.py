from types import SimpleNamespace

from mcp_service.retriever_service import LegalRetrieverService
from mcp_service.tool_registry import ToolRegistry


def _registry(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(
        '[{"name":"search_civil_law","description":"civil","domain":"civil",'
        '"collection":"civil_law","parameters":{"type":"object"}},'
        '{"name":"search_labor_law","description":"labor","domain":"labor",'
        '"collection":"labor_law","parameters":{"type":"object"}}]',
        encoding="utf-8",
    )
    return ToolRegistry(path)


def test_tool_names_resolve_to_different_collections(tmp_path):
    service = LegalRetrieverService.__new__(LegalRetrieverService)
    service.registry = _registry(tmp_path)

    assert service.collection_for_tool("search_civil_law") == "civil_law"
    assert service.collection_for_tool("search_labor_law") == "labor_law"


def test_search_uses_collection_selected_by_tool_contract(tmp_path, monkeypatch):
    service = LegalRetrieverService.__new__(LegalRetrieverService)
    service.registry = _registry(tmp_path)
    calls = []

    class FakeRetriever:
        def invoke(self, query):
            return [SimpleNamespace(page_content=f"result:{query}")]

    def fake_get_retriever(collection):
        calls.append(collection)
        return FakeRetriever()

    monkeypatch.setattr(service, "_get_retriever", fake_get_retriever)

    assert service.search("search_civil_law", "民法问题") == "result:民法问题"
    assert service.search("search_labor_law", "劳动问题") == "result:劳动问题"
    assert calls == ["civil_law", "labor_law"]


def test_search_rejects_blank_query(tmp_path):
    service = LegalRetrieverService.__new__(LegalRetrieverService)
    service.registry = _registry(tmp_path)

    try:
        service.search("search_civil_law", "   ")
    except ValueError as exc:
        assert str(exc) == "query 不能为空"
    else:
        raise AssertionError("blank query should be rejected")
