from knowledge.query_rewrite import rewrite_legal_query


def test_civil_loan_query_expands_legal_terms():
    text = rewrite_legal_query("借钱到期不还怎么办？", "civil")
    assert "借款合同" in text
    assert "返还借款" in text


def test_civil_water_damage_query_expands_neighbor_terms():
    text = rewrite_legal_query("楼上漏水把我家泡了，可以赔偿吗？", "civil")
    assert "侵权责任" in text
    assert "相邻关系" in text


def test_non_civil_query_is_unchanged():
    question = "公司不给工资怎么办？"
    assert rewrite_legal_query(question, "labor") == question
