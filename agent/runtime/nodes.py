"""LangGraph nodes for the first production-oriented Agent Runtime."""

from __future__ import annotations

from typing import Any, Callable

from knowledge.retriever import LegalRetriever
from mcp_service.tool_registry import ToolRegistry
from mcp_service.retriever_service import build_default_service

from .state import AgentState
from ..router import HybridRouter


_ROUTER = HybridRouter()


def _trace(state: AgentState, node: str, **extra: Any) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, **extra}]


def intent_analysis(state: AgentState) -> AgentState:
    question = state["question"]
    route = _ROUTER.route(question)
    intent = "legal_retrieval" if route.confidence > 0 else "legal_consultation"
    return {
        "domain": route.domain,
        "intent": intent,
        "intent_confidence": route.confidence,
        "route_candidates": route.candidates,
        "trace": _trace(state, "intent_analysis", method=route.method, domain=route.domain),
    }


def task_planning(state: AgentState) -> AgentState:
    plan = ["route_to_domain_tool", "retrieve_legal_basis", "generate_answer", "verify_answer"]
    return {
        "plan": plan,
        "current_step": 0,
        "retry_count": state.get("retry_count", 0),
        "trace": _trace(state, "task_planning", steps=plan),
    }


def tool_decision(state: AgentState) -> AgentState:
    domain = state.get("domain", "civil")
    tool_name = f"search_{domain}_law"
    return {
        "tool_name": tool_name,
        "tool_arguments": {"query": state["question"], "limit": 5},
        "trace": _trace(state, "tool_decision", tool=tool_name),
    }


def tool_execution(state: AgentState, service=None) -> AgentState:
    service = service or build_default_service()
    tool_name = state["tool_name"]
    arguments = state.get("tool_arguments", {})
    try:
        result = service.search(tool_name, arguments["query"], int(arguments.get("limit", 5)))
        return {
            "tool_result": result,
            "error": None,
            "trace": _trace(state, "tool_execution", tool=tool_name, status="success"),
        }
    except Exception as exc:
        return {
            "tool_result": "",
            "error": str(exc),
            "trace": _trace(state, "tool_execution", tool=tool_name, status="error", error=str(exc)),
        }


def retrieval(state: AgentState) -> AgentState:
    # The MCP/Retriever service currently returns a rendered result. Keep the
    # raw result in state and expose a structured citation list for downstream nodes.
    text = state.get("tool_result", "")
    citations: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        if block.startswith("[") and "]\n" in block:
            header, content = block.split("]\n", 1)
            citations.append({"reference": header[1:], "content": content})
    return {
        "retrieved_documents": citations,
        "citations": citations,
        "trace": _trace(state, "retrieval", documents=len(citations)),
    }


def generation(state: AgentState, answer_generator: Callable[[str, str, list[dict[str, Any]]], str] | None = None) -> AgentState:
    question = state["question"]
    evidence = state.get("tool_result", "")
    citations = state.get("citations", [])
    if answer_generator is not None:
        answer = answer_generator(question, evidence, citations)
    elif evidence:
        answer = "根据检索到的法律依据，建议结合以下条款进一步判断：\n\n" + evidence
    else:
        answer = "暂未检索到足够的法律依据，无法给出可靠结论。"
    return {
        "answer": answer,
        "trace": _trace(state, "generation", answer_length=len(answer)),
    }


def verification(state: AgentState) -> AgentState:
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    ok = bool(answer.strip()) and bool(citations)
    verification = {
        "passed": ok,
        "citation_count": len(citations),
        "reason": "answer_and_citations_present" if ok else "missing_answer_or_citations",
    }
    return {
        "verification": verification,
        "trace": _trace(state, "verification", **verification),
    }


def route_after_tool(state: AgentState) -> str:
    if state.get("error"):
        return "retry_or_end"
    return "retrieval"


def route_after_verification(state: AgentState) -> str:
    if state.get("verification", {}).get("passed"):
        return "end"
    if state.get("retry_count", 0) < 1:
        return "retry"
    return "end"


def retry_plan(state: AgentState) -> AgentState:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "plan": ["retry_retrieval", "generate_answer", "verify_answer"],
        "tool_arguments": {"query": state["question"], "limit": 8},
        "trace": _trace(state, "retry_plan", retry_count=state.get("retry_count", 0) + 1),
    }
