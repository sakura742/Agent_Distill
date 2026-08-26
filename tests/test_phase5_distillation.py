import json
from pathlib import Path

from distill.trajectory_schema import ToolCall, AgentStep, AgentTrajectory, validate_trajectory
from distill.convert_dataset import convert


def test_valid_trajectory_schema():
    payload = AgentTrajectory(
        user_query="公司拖欠工资怎么办？",
        intent="labor_consultation",
        domain="labor",
        steps=[AgentStep(1, "选择劳动法检索工具", ToolCall("search_labor_law", {"query": "拖欠工资"}))],
    ).to_dict()
    assert validate_trajectory(payload) == []


def test_invalid_trajectory_is_reported():
    errors = validate_trajectory({"user_query": "x", "intent": "labor", "steps": []})
    assert any("non-empty list" in error for error in errors)


def test_convert_trajectory_to_sft(tmp_path: Path):
    source = tmp_path / "traces.jsonl"
    target = tmp_path / "train.jsonl"
    source.write_text(json.dumps({
        "user_query": "拖欠工资怎么办？",
        "intent": "labor_consultation",
        "domain": "labor",
        "steps": [{"step": 1, "action": "检索劳动法", "tool_call": {"name": "search_labor_law", "arguments": {"query": "拖欠工资"}}, "observation": [{"article": "第八十五条"}]}],
        "final_answer": "可以先申请劳动仲裁。",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    total, written = convert(source, target)
    assert (total, written) == (1, 1)
    row = json.loads(target.read_text(encoding="utf-8"))
    assert row["input"] == "拖欠工资怎么办？"
    assert "search_labor_law" in row["output"]
    assert "第八十五条" in row["output"]
