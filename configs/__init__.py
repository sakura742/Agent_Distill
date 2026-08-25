"""Configuration package for Agent_Distill.

Exposes a single module-level ``settings`` object (see ``configs.settings``)
that centralizes every path / secret that used to be hardcoded across the
codebase (Windows absolute paths, the DeepSeek API key, etc.).
"""

from configs.settings import settings

__all__ = ["settings"]
