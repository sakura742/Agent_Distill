from evaluation.retrieval_metrics import evaluate_retrieval, mrr, recall_at_k


def test_recall_at_k_and_mrr():
    retrieved = ["a", "b", "c"]
    relevant = {"b"}
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert mrr(retrieved, relevant) == 0.5


def test_evaluate_retrieval():
    result = evaluate_retrieval(
        [
            {"retrieved_ids": ["x", "a"], "relevant_ids": ["a"]},
            {"retrieved_ids": ["b", "c"], "relevant_ids": ["z"]},
        ],
        k=2,
    )
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.5
