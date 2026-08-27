from pathlib import Path

import pytest

from distill.runtime_teacher import _openai_tools, collect_trajectory, load_tool_contract
from distill.trajectory_schema import AgentTrajectory


def test_openai_tool_contract_conversion():
    tools = [{
        "name": "search_labor_law",
        "description": "search labor law",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }]
    converted = _openai_tools(tools)
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "search_labor_law"


def test_load_tool_contract(tmp_path: Path):
    path = tmp_path / "tools.json"
    path.write_text('{"tools": [{"name": "search_civil_law"}]}', encoding="utf-8")
    assert load_tool_contract(path)[0]["name"] == "search_civil_law"


def test_collect_trajectory_executes_real_executor(monkeypatch):
    class Message:
        def __init__(self, calls=None, content=""):
            self.tool_calls = calls
            self.content = content

    class Call:
        id = "call-1"
        function = type("Function", (), {
            "name": "search_labor_law",
            "arguments": '{"query":"拖欠工资"}',
        })()

        def model_dump(self):
            return {"id": self.id, "type": "function", "function": {
                "name": self.function.name, "arguments": self.function.arguments,
            }}

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return type("Response", (), {"choices": [type("Choice", (), {"message": Message([Call()])})()]})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": Message([], "基于劳动法检索结果给出回答")})()]})()

    completions = Completions()
    fake_client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    monkeypatch.setattr("distill.runtime_teacher.OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr("distill.runtime_teacher.settings.deepseek_api_key", "test-key")

    observed = []

    def execute(name, arguments):
        observed.append((name, arguments))
        return "[劳动法/第1条]\n工资应当按时支付。"

    result = collect_trajectory(
        "公司拖欠工资怎么办？",
        [{"name": "search_labor_law", "parameters": {"type": "object"}}],
        execute,
    )
    assert isinstance(result, AgentTrajectory)
    assert observed == [("search_labor_law", {"query": "拖欠工资"})]
    assert result.domain == "labor"
    assert result.steps[0].observation
