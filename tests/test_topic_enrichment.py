from knowledge.topic_enrichment import chapter_topics, enriched_retrieval_text


def test_tort_chapter_gets_general_liability_topics():
    topics = chapter_topics("第七章 侵权责任")
    assert "一般侵权" in topics
    assert "过错责任" in topics
    assert "损害赔偿" in topics


def test_enriched_text_keeps_original_article_text():
    text = enriched_retrieval_text(
        law_name="中华人民共和国民法典",
        article="一千一百六十五",
        chapter="第七章 侵权责任",
        text="行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。",
    )
    assert "第一千一百六十五条" in text
    assert "一般侵权" in text
    assert "行为人因过错侵害" in text
