from agent.router import HybridRouter, UNKNOWN_DOMAIN


def test_non_legal_query_abstains_without_embedding(monkeypatch):
    router = HybridRouter()
    monkeypatch.setattr(router, "_semantic_scores", lambda _: {"civil": 0.31, "labor": 0.30})
    result = router.route("你好，今天天气不错，你吃了吗？")
    assert result.domain == UNKNOWN_DOMAIN
    assert result.confidence == 0.0
    assert result.method == "embedding_abstain"


def test_rule_match_still_routes_to_legal_domain():
    result = HybridRouter().route("公司不给我加班费怎么办？")
    assert result.domain == "labor"
    assert result.confidence > 0
