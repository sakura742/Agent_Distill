"""Tool contract and registry for the MCP service.

Phase 2 goal: make ``tools_config.json`` the single source of truth for
public tool metadata while keeping execution logic in Python. The registry
validates tool definitions at startup and exposes an immutable lookup API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.exceptions import ConfigurationError


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    domain: str
    collection: str


class ToolRegistry:
    """Load and validate the project's MCP tool contract exactly once."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self._tools = self._load()

    def _load(self) -> dict[str, ToolDefinition]:
        if not self.config_path.exists():
            raise ConfigurationError(f"工具配置不存在: {self.config_path}")

        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"无法读取工具配置: {self.config_path}") from exc

        if not isinstance(raw, list) or not raw:
            raise ConfigurationError("工具配置必须是非空数组")

        result: dict[str, ToolDefinition] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ConfigurationError("每个工具定义必须是对象")
            name = item.get("name")
            description = item.get("description")
            parameters = item.get("parameters")
            domain = item.get("domain")
            collection = item.get("collection")
            if not all(isinstance(value, str) and value for value in (name, description, domain, collection)):
                raise ConfigurationError(f"工具 {name!r} 缺少 name/description/domain/collection")
            if not isinstance(parameters, dict):
                raise ConfigurationError(f"工具 {name!r} 的 parameters 必须是对象")
            if name in result:
                raise ConfigurationError(f"工具名称重复: {name}")
            result[name] = ToolDefinition(name, description, parameters, domain, collection)
        return result

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ConfigurationError(f"未注册 MCP 工具: {name}") from exc

    def all(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools.values())
