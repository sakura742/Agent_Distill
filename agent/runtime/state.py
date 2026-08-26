"""Shared state for the LangGraph Agent Runtime."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    conversation_id: str
    domain: str
    intent: str
    intent_confidence: float
    route_candidates: list[dict[str, Any]]
    plan: list[str]
    current_step: int
    tool_name: str | None
    tool_arguments: dict[str, Any]
    tool_result: str
    retrieved_documents: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    answer: str
    verification: dict[str, Any]
    retry_count: int
    error: str | None
    trace: list[dict[str, Any]]
