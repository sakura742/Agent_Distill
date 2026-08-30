"""Stateful LangGraph orchestration for legal consultation."""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from .nodes import (
    generation,
    intent_analysis,
    retrieval,
    retry_plan,
    route_after_tool,
    route_after_verification,
    task_planning,
    tool_decision,
    tool_execution,
    verification,
)
from .state import AgentState
from .tool_executor import DirectToolExecutor


def build_legal_agent_graph(
    *,
    tool_service=None,
    tool_executor=None,
    answer_generator: Callable | None = None,
    decision_generator: Callable | None = None,
):
    """Build the legal Agent graph with injectable decision and answer models.

    ``decision_generator`` is intentionally optional. Existing Phase 2-4 callers
    therefore keep deterministic routing/tool behavior, while Phase 5 can inject
    the Decision LoRA and measure whether its action is actually consumed.
    """
    if tool_executor is None:
        tool_executor = DirectToolExecutor(tool_service)

    graph = StateGraph(AgentState)
    graph.add_node("intent_analysis", intent_analysis)
    graph.add_node("task_planning", task_planning)
    graph.add_node("tool_decision", lambda state: tool_decision(state, decision_generator))
    graph.add_node("tool_execution", lambda state: tool_execution(state, tool_executor))
    graph.add_node("retrieval", retrieval)
    graph.add_node("generation", lambda state: generation(state, answer_generator))
    graph.add_node("verification", verification)
    graph.add_node("retry_plan", retry_plan)

    graph.add_edge(START, "intent_analysis")
    graph.add_edge("intent_analysis", "task_planning")
    graph.add_edge("task_planning", "tool_decision")
    graph.add_edge("tool_decision", "tool_execution")
    graph.add_conditional_edges(
        "tool_execution",
        route_after_tool,
        {"retrieval": "retrieval", "retry_or_end": "retry_plan"},
    )
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", "verification")
    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"end": END, "retry": "retry_plan"},
    )
    graph.add_edge("retry_plan", "tool_execution")

    return graph.compile()
