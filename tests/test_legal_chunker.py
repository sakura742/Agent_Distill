from knowledge.legal_chunker import split_legal_text


def test_split_preserves_article_and_chapter_metadata():
    text = "第一章 总则\n第一条 为了规范劳动关系，制定本法。\n第二条 本法适用于中华人民共和国境内的企业。"
    chunks = split_legal_text(
        text,
        domain="labor",
        law_name="中华人民共和国劳动法",
        source="labor_law.pdf",
        page=2,
    )
    assert len(chunks) == 2
    assert chunks[0].article == "一"
    assert chunks[0].chapter == "第一章 总则"
    assert chunks[0].page == 2
    assert chunks[0].domain == "labor"
    assert chunks[0].to_metadata()["law_name"] == "中华人民共和国劳动法"


def test_long_article_is_split_without_losing_article_metadata():
    text = "第十条 " + ("这是条款内容。" * 400)
    chunks = split_legal_text(
        text,
        domain="civil",
        law_name="中华人民共和国民法典",
        source="minfa.pdf",
        page=10,
        max_chars=1000,
    )
    assert len(chunks) > 1
    assert all(chunk.article == "十" for chunk in chunks)
