from evaluation.benchmark import evaluate
from evaluation.metrics import (
    accuracy,
    argument_accuracy,
    citation_accuracy,
    citation_precision,
    citation_recall,
    interruption_error,
    mrr,
    recall_at_k,
    workflow_success,
)


def test_accuracy():
    assert accuracy(["labor", "civil"], ["labor", "labor"]) == 0.5


def test_retrieval_metrics():
    retrieved = [["a", "b", "c"], ["x", "y"]]
    relevant = [["b"], ["z"]]
    assert recall_at_k(retrieved, relevant, 5) == 0.5
    assert mrr(retrieved, relevant) == 0.5


def test_citation_metrics():
    assert citation_precision(["a", "b"], ["b", "c"]) == 0.5
    assert citation_recall(["a", "b"], ["b", "c"]) == 0.5
    assert citation_accuracy(["a"], ["a"]) == 1.0


def test_argument_accuracy_is_per_parameter():
    assert argument_accuracy({"query": "q", "limit": 5}, {"query": "q", "limit": 5}) == 1.0
    assert argument_accuracy({"query": "q", "limit": 8}, {"query": "q", "limit": 5}) == 0.5


def test_workflow_and_interruption_metrics():
    trace = [{"node": n} for n in [
        "intent_analysis", "task_planning", "tool_decision",
        "tool_execution", "retrieval", "generation", "verification",
    ]]
    assert workflow_success(trace, {"passed": True}) == 1.0
    assert interruption_error(trace, {"passed": True}) == 0.0
    assert interruption_error(trace[:2], {"passed": False}) == 1.0


def test_benchmark_evaluator_scores_all_core_tracks():
    rows = [
        {"category": "routing", "prediction": "labor", "gold": "labor"},
        {"category": "retrieval", "retrieved": ["doc1"], "gold": ["doc1"]},
        {"category": "tool_calling", "prediction": {"name": "search_labor_law", "arguments": {"limit": 5}}, "gold": {"name": "search_labor_law", "arguments": {"limit": 5}}},
        {"category": "workflow", "trace": [{"node": n} for n in ["intent_analysis", "task_planning", "tool_decision", "tool_execution", "retrieval", "generation", "verification"]], "verification": {"passed": True}},
        {"category": "answer", "prediction": "abc", "gold": "abc"},
        {"category": "answer_citation", "predicted": ["doc1"], "gold": ["doc1"]},
    ]
    result = evaluate(rows)
    assert result["routing_accuracy"] == 1.0
    assert result["retrieval_recall_at_5"] == 1.0
    assert result["tool_selection_accuracy"] == 1.0
    assert result["tool_argument_accuracy"] == 1.0
    assert result["workflow_success_rate"] == 1.0
    assert result["interruption_error_rate"] == 0.0
    assert result["citation_accuracy"] == 1.0
