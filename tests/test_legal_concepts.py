from knowledge.legal_concepts import article_concepts
from knowledge.legal_chunker import documents_from_pdf_pages


def test_civil_article_concepts_are_auditable():
    assert "一般侵权责任" in article_concepts("civil", "1165")
    assert "财产损失" in article_concepts("civil", "1184")
    assert article_concepts("labor", "1165") == ()


def test_article_concepts_are_added_only_to_retrieval_text():
    docs = documents_from_pdf_pages(
        ["第一千一百六十五条 行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。"],
        domain="civil",
        law_name="中华人民共和国民法典",
        source="minfa.pdf",
    )
    assert len(docs) == 1
    doc = docs[0]
    assert "一般侵权责任" in doc.page_content
    assert doc.metadata["original_text"].startswith("第一千一百六十五条")
    assert doc.metadata["legal_concepts"]
