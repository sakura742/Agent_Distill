from evaluation.metrics import accuracy, recall_at_k, mrr, citation_precision, citation_recall


def test_accuracy():
    assert accuracy(["labor", "civil"], ["labor", "labor"]) == 0.5


def test_retrieval_metrics():
    retrieved=[["a","b","c"],["x","y"]]
    relevant=[["b"],["z"]]
    assert recall_at_k(retrieved,relevant,5)==0.5
    assert mrr(retrieved,relevant)==0.5


def test_citation_metrics():
    assert citation_precision(["a","b"],["b","c"])==0.5
    assert citation_recall(["a","b"],["b","c"])==0.5
