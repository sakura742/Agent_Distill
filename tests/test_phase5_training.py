from pathlib import Path

from distill.format_sft import trajectory_to_messages
from distill.training_config import LoRAConfig


def test_trajectory_to_messages_preserves_tool_call_and_observation():
    item = {
        "user_query": "拖欠工资怎么办？",
        "steps": [
            {
                "step": 1,
                "action": "识别为劳动法问题",
                "tool_call": {"name": "search_labor_law", "arguments": {"query": "拖欠工资"}},
                "observation": "劳动合同法相关条款",
            }
        ],
        "final_answer": "可以依据相关规定维权。",
    }
    messages = trajectory_to_messages(item)
    assert messages[1]["role"] == "user"
    assert any(m["role"] == "tool" and "劳动合同法" in m["content"] for m in messages)
    assert any("search_labor_law" in m["content"] for m in messages)
    assert messages[-1]["content"] == "可以依据相关规定维权。"


def test_lora_config_validation_rejects_missing_model(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    config = LoRAConfig(tmp_path / "missing", train_file, tmp_path / "out")
    try:
        config.validate()
    except FileNotFoundError as exc:
        assert "Base model" in str(exc)
    else:
        raise AssertionError("missing base model should fail validation")
