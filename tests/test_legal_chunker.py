from knowledge.legal_chunker import split_legal_text, split_legal_text_with_pages


def test_split_preserves_article_and_chapter_metadata():
    text = "第一章 总则\n第一条 为了规范劳动关系，制定本法。\n第二条 本法适用于中华人民共和国境内的企业。"
    chunks = split_legal_text(text, domain="labor", law_name="中华人民共和国劳动法", source="labor_law.pdf", page=2)
    assert len(chunks) == 2
    assert chunks[0].article == "一"
    assert chunks[0].chapter == "第一章 总则"
    assert chunks[0].page == 2
    assert chunks[0].domain == "labor"
    assert chunks[0].to_metadata()["law_name"] == "中华人民共和国劳动法"


def test_long_article_is_split_without_losing_article_metadata():
    text = "第十条 " + ("这是条款内容。" * 400)
    chunks = split_legal_text(text, domain="civil", law_name="中华人民共和国民法典", source="minfa.pdf", page=10, max_chars=1000)
    assert len(chunks) > 1
    assert all(chunk.article == "十" for chunk in chunks)


def test_cross_page_article_is_not_truncated():
    pages = [
        "第三十条 用人单位应当按照约定及时足额支付劳动报酬；",
        "逾期不支付的，劳动者可以依法主张相应权利。\n第三十一条 用人单位应当严格执行劳动定额标准。",
    ]
    chunks = split_legal_text_with_pages(pages, domain="labor", law_name="劳动法", source="labor_law.pdf")
    article30 = [c for c in chunks if c.article == "三十"]
    assert len(article30) == 1
    assert "及时足额支付劳动报酬" in article30[0].text
    assert "逾期不支付" in article30[0].text
    assert article30[0].page == 1


def test_adjacent_articles_remain_independent():
    pages = ["第一百条 A内容\n第一百零一条 B内容\n第一百零二条 C内容"]
    chunks = split_legal_text_with_pages(pages, domain="labor", law_name="劳动法", source="labor_law.pdf")
    assert [c.article for c in chunks] == ["一百", "一百零一", "一百零二"]
