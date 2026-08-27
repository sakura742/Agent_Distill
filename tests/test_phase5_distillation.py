from pathlib import Path


def test_phase5_configuration_is_separate_from_phase2_model():
    from configs.settings import settings

    assert settings.qwen35_model_path
    assert settings.qwen35_model_path != settings.base_model_path
    assert settings.qwen35_lora_output_dir != settings.lora_output_dir


def test_hard_example_rules():
    from distill.hard_mining import is_hard

    assert is_hard({"verification": {"passed": False}})
    assert is_hard({"retry_count": 1})
    assert is_hard({"intent_confidence": 0.2})
    assert is_hard({"citations": []})
    assert not is_hard({
        "verification": {"passed": True},
        "retry_count": 0,
        "intent_confidence": 0.9,
        "tool": {"name": "search_civil_law", "arguments": {"query": "借款"}},
        "citations": [{"reference": "民法典"}],
    })


def test_trajectory_serializer_contains_observable_agent_behavior(tmp_path: Path):
    from distill.train_phase5 import _trajectory_to_text

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(m["role"] + ":" + m["content"] for m in messages)

    text = _trajectory_to_text({
        "question": "借款不还怎么办？",
        "domain": "civil",
        "intent": "legal_retrieval",
        "plan": ["route_to_domain_tool"],
        "tool": {"name": "search_civil_law", "arguments": {"query": "民间借贷"}},
        "citations": [{"reference": "民法典"}],
        "answer": "建议先固定证据。",
    }, Tokenizer())
    assert "search_civil_law" in text
    assert "民间借贷" in text
    assert "建议先固定证据" in text
