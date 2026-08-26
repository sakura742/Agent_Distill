import json

import pytest

from app.exceptions import ConfigurationError
from mcp_service.tool_registry import ToolRegistry


def test_registry_loads_current_tool_contract(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([
        {
            "name": "search_labor_law",
            "description": "labor",
            "domain": "labor",
            "collection": "labor_law",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]), encoding="utf-8")

    registry = ToolRegistry(path)
    tool = registry.get("search_labor_law")

    assert tool.domain == "labor"
    assert tool.collection == "labor_law"
    assert tool.parameters["type"] == "object"


def test_registry_rejects_duplicate_names(tmp_path):
    item = {
        "name": "search_law",
        "description": "law",
        "domain": "labor",
        "collection": "labor_law",
        "parameters": {"type": "object"},
    }
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([item, item]), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="工具名称重复"):
        ToolRegistry(path)


def test_registry_rejects_missing_contract_file(tmp_path):
    with pytest.raises(ConfigurationError, match="工具配置不存在"):
        ToolRegistry(tmp_path / "missing.json")
