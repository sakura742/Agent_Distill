from pathlib import Path

from evaluation.retrieval_diagnostics import run


def test_candidate_pool_diagnostics_marks_gold_reachability(tmp_path: Path):
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        '{"id":"x","question":"买卖合同逾期交货怎么办？","domain":"civil","gold_references":["中华人民共和国民法典 五百七十七"]}\n',
        encoding="utf-8",
    )

    class FakeRetriever:
        def search(self, *args, **kwargs):
            from types import SimpleNamespace
            return [
                SimpleNamespace(metadata={"law_name":"中华人民共和国民法典", "article":"五百七十八"}, score=0.9),
                SimpleNamespace(metadata={"law_name":"中华人民共和国民法典", "article":"五百七十七"}, score=0.8),
            ]

    import evaluation.retrieval_diagnostics as module
    original = module.get_retriever
    module.get_retriever = lambda *args, **kwargs: FakeRetriever()
    try:
        report = run(benchmark, candidate_k=2)
    finally:
        module.get_retriever = original

    assert report["details"][0]["gold_in_candidate_pool"] is True
    assert report["details"][0]["gold_ranks"]["中华人民共和国民法典 五百七十七"] == 2
