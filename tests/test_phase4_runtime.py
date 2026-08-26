from agent.router import HybridRouter
from agent.runtime.graph import build_legal_agent_graph


class FakeToolService:
    def search(self, tool_name, query, limit=5):
        assert tool_name in {"search_labor_law", "search_civil_law"}
        return "[劳动合同法 第三十条 | labor_law.pdf p.1]\n用人单位应当按照约定及时足额支付劳动报酬。"


def test_hybrid_router_prefers_rule_match():
    result = HybridRouter().route("公司突然辞退我，还拖欠工资怎么办？")
    assert result.domain == "labor"
    assert result.method == "rule"
    assert result.confidence > 0.5


def test_graph_executes_stateful_workflow():
    graph = build_legal_agent_graph(tool_service=FakeToolService())
    result = graph.invoke({"question": "公司拖欠我的工资怎么办？", "conversation_id": "test-1"})

    assert result["domain"] == "labor"
    assert result["tool_name"] == "search_labor_law"
    assert result["verification"]["passed"] is True
    assert result["citations"]
    assert [item["node"] for item in result["trace"]] == [
        "intent_analysis",
        "task_planning",
        "tool_decision",
        "tool_execution",
        "retrieval",
        "generation",
        "verification",
    ]
