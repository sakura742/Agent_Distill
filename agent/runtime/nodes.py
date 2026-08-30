"""LangGraph nodes for the legal Agent Runtime."""
from __future__ import annotations
import re
from typing import Any, Callable
from .state import AgentState
from ..router import HybridRouter, UNKNOWN_DOMAIN
from .tool_executor import DirectToolExecutor

_ROUTER = HybridRouter()


def _trace(state: AgentState, node: str, **extra: Any) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"node": node, **extra}]


def intent_analysis(state: AgentState) -> AgentState:
    route = _ROUTER.route(state["question"])
    intent = "non_legal" if route.domain == UNKNOWN_DOMAIN else "legal_retrieval"
    return {"domain": route.domain, "intent": intent, "intent_confidence": route.confidence,
            "route_candidates": route.candidates,
            "trace": _trace(state, "intent_analysis", method=route.method, domain=route.domain)}


def route_after_intent(state: AgentState) -> str:
    return "non_legal" if state.get("domain") == UNKNOWN_DOMAIN else "legal"


def task_planning(state: AgentState) -> AgentState:
    plan = ["route_to_domain_tool", "retrieve_legal_basis", "generate_answer", "verify_answer"]
    return {"plan": plan, "current_step": 0, "retry_count": state.get("retry_count", 0),
            "trace": _trace(state, "task_planning", steps=plan)}


def tool_decision(state: AgentState, decision_generator: Callable[[str], dict[str, Any] | None] | None = None) -> AgentState:
    decision = decision_generator(state["question"]) if decision_generator else None
    if decision:
        tool = decision.get("tool") or {}
        domain = str(decision.get("domain") or state.get("domain", "civil"))
        name = str(tool.get("name") or "")
        if name in {"search_labor_law", "search_civil_law"} and name.endswith(f"{domain}_law"):
            args = tool.get("arguments") or {}
            args = {"query": str(args.get("query") or state["question"]), "limit": int(args.get("limit", 5))}
            return {"domain": domain, "intent": decision.get("intent", state.get("intent")), "tool_name": name,
                    "tool_arguments": args, "trace": _trace(state, "tool_decision", tool=name, method="model")}
    domain = state.get("domain", UNKNOWN_DOMAIN)
    if domain == UNKNOWN_DOMAIN:
        return {"tool_name": None, "tool_arguments": {}, "trace": _trace(state, "tool_decision", tool=None, method="abstain")}
    name = f"search_{domain}_law"
    return {"tool_name": name, "tool_arguments": {"query": state["question"], "limit": 5},
            "trace": _trace(state, "tool_decision", tool=name, method="fallback")}


def tool_execution(state: AgentState, executor=None) -> AgentState:
    if not state.get("tool_name"):
        return {"tool_result": "", "error": None,
                "trace": _trace(state, "tool_execution", tool=None, status="skipped")}
    executor = executor or DirectToolExecutor()
    try:
        return {"tool_result": executor.execute(state["tool_name"], state.get("tool_arguments", {})),
                "error": None, "trace": _trace(state, "tool_execution", tool=state["tool_name"], status="success")}
    except Exception as exc:
        return {"tool_result": "", "error": str(exc),
                "trace": _trace(state, "tool_execution", tool=state["tool_name"], status="error", error=str(exc))}


def _split_articles(content: str) -> list[str]:
    """Split a retriever chunk containing adjacent Chinese law articles."""
    matches = list(re.finditer(r"(?m)(?=^第[一二三四五六七八九十百千万亿零〇两]+条\s*$)", content))
    if not matches:
        return [content.strip()] if content.strip() else []
    chunks: list[str] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chunk = content[match.start():end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _article_number(text: str) -> str | None:
    m = re.search(r"第([一二三四五六七八九十百千万亿零〇两]+)条", text.strip())
    return m.group(1) if m else None


def _reference_article_number(reference: str) -> str | None:
    """Extract an article number from a reference header when legacy retrievers omit it from content."""
    return _article_number(reference) or (re.search(r"(?:^|\s)(\d+)\s*(?:条)?(?:\s*\||$)", reference).group(1)
                                           if re.search(r"(?:^|\s)(\d+)\s*(?:条)?(?:\s*\||$)", reference) else None)


def _parse_retrieval(tool_result: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for block in tool_result.split("\n\n"):
        if not block.startswith("[") or "]\n" not in block:
            continue
        header, content = block.split("]\n", 1)
        reference = header[1:].strip()
        score = None
        score_match = re.search(r"\s*\|\s*score=([0-9]*\.?[0-9]+)\s*$", reference)
        if score_match:
            score = float(score_match.group(1))
            reference = reference[:score_match.start()].rstrip()
        for article in _split_articles(content):
            number = _article_number(article)
            ref = reference
            if number and " | " in reference:
                _, suffix = reference.split(" | ", 1)
                ref = f"{number} | {suffix}"
            elif number:
                ref = number
            doc = {"reference": ref, "content": article}
            if score is not None:
                doc["score"] = score
            docs.append(doc)
    return docs


def retrieval(state: AgentState) -> AgentState:
    docs = _parse_retrieval(state.get("tool_result", ""))
    return {"retrieved_documents": docs, "citations": [],
            "trace": _trace(state, "retrieval", documents=len(docs), scores=[d.get("score") for d in docs if "score" in d])}


def _select_citations(answer: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only retrieved articles explicitly used by the answer."""
    if not answer.strip():
        return []
    selected: list[dict[str, Any]] = []
    for doc in docs:
        ref = str(doc.get("reference", ""))
        article = _article_number(str(doc.get("content", ""))) or _reference_article_number(ref)
        if (ref and ref in answer) or (article and article in answer):
            selected.append(doc)
    return selected


def _answer_article_numbers(answer: str) -> set[str]:
    """Return article numbers explicitly mentioned by the generated answer."""
    return set(re.findall(r"第([一二三四五六七八九十百千万亿零〇两]+)条", answer))


def generation(state: AgentState, answer_generator: Callable[[str, str, list[dict[str, Any]]], str] | None = None) -> AgentState:
    q = state["question"]
    evidence = state.get("tool_result", "")
    docs = state.get("retrieved_documents", [])
    if state.get("domain") == UNKNOWN_DOMAIN and answer_generator is None:
        answer = "您好！如果您有具体的法律问题，可以告诉我事情经过、涉及的合同或法律关系，我可以帮您分析。"
    else:
        answer = answer_generator(q, evidence, docs) if answer_generator else (
            "根据检索到的法律依据，建议结合以下条款进一步判断：\n\n" + evidence
            if evidence else "暂未检索到足够的法律依据，无法给出可靠结论。")
    citations = _select_citations(answer, docs) if state.get("domain") != UNKNOWN_DOMAIN else []
    return {"answer": answer, "citations": citations,
            "trace": _trace(state, "generation", answer_length=len(answer), citations=len(citations))}


def verification(state: AgentState) -> AgentState:
    answer = state.get("answer", "")
    docs = state.get("retrieved_documents", [])
    citations = state.get("citations", [])
    non_legal = state.get("domain") == UNKNOWN_DOMAIN
    if non_legal:
        ok = bool(answer.strip()) and not citations and not state.get("tool_name")
        reason = "non_legal_answer_present" if ok else "non_legal_routed_to_tool_or_has_citations"
    else:
        cited_refs = {str(d.get("reference", "")) for d in citations}
        retrieved_refs = {str(d.get("reference", "")) for d in docs}
        valid_citations = cited_refs.issubset(retrieved_refs)
        citation_numbers = {_article_number(str(d.get("content", ""))) or _reference_article_number(str(d.get("reference", ""))) for d in citations}
        mentioned_numbers = _answer_article_numbers(answer)
        citation_mentions = {n for n in citation_numbers if n}
        answer_mentions_citation = bool(citations) and citation_mentions.intersection(mentioned_numbers)
        unsupported_answer_citations = mentioned_numbers - citation_mentions

        # If the answer names an article that was not among the selected
        # citations, the answer is not grounded in the retrieved evidence.
        # This catches cases such as a labor-law retrieval followed by an
        # answer citing unrelated Civil Code articles from model memory.
        grounded = not unsupported_answer_citations

        # A score of zero is explicitly treated as "unknown relevance" rather
        # than evidence of relevance. Legacy fake retrievers without a score
        # remain compatible with the Phase 4 tests.
        scored_docs = [d for d in citations if "score" in d]
        relevance_known = not scored_docs or any(float(d.get("score", 0.0)) > 0.0 for d in scored_docs)

        ok = (
            bool(answer.strip()) and bool(citations) and valid_citations
            and answer_mentions_citation and grounded and relevance_known
        )
        if not grounded:
            reason = "answer_cites_unretrieved_articles"
        elif not relevance_known:
            reason = "citation_relevance_unknown"
        else:
            reason = "answer_and_used_citations_valid" if ok else "missing_invalid_or_unsubstantiated_citations"
    v = {"passed": ok, "citation_count": len(citations), "retrieved_count": len(docs), "reason": reason}
    return {"verification": v, "trace": _trace(state, "verification", **v)}


def route_after_tool(state: AgentState) -> str:
    return "retry_or_end" if state.get("error") else "retrieval"


def route_after_verification(state: AgentState) -> str:
    return "end" if state.get("verification", {}).get("passed") else ("retry" if state.get("retry_count", 0) < 1 else "end")


def retry_plan(state: AgentState) -> AgentState:
    return {"retry_count": state.get("retry_count", 0) + 1,
            "trace": _trace(state, "retry_plan", retry_count=state.get("retry_count", 0) + 1)}
