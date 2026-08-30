from knowledge.retriever import _distance_to_score


def test_l2_score_is_monotonic_and_nonzero():
    assert _distance_to_score(0.0, "l2") == 1.0
    assert _distance_to_score(1.0, "l2") == 0.5
    assert _distance_to_score(3.0, "l2") == 0.25
    assert _distance_to_score(3.0, "l2") > 0.0


def test_cosine_score_uses_one_minus_distance():
    assert _distance_to_score(0.1, "cosine") == 0.9
    assert _distance_to_score(1.4, "cosine") == 0.0
