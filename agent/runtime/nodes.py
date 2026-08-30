"""LangGraph nodes for the legal Agent Runtime."""
from __future__ import annotations
from typing import Any, Callable
from .state import AgentState
from ..router import HybridRouter
from .tool_executor import DirectToolExecutor
_ROUTER=HybridRouter()
def _trace(state:AgentState,node:str,**extra:Any)->list[dict[str,Any]]: return [*state.get("trace",[]),{"node":node,**extra}]
def intent_analysis(state:AgentState)->AgentState:
    route=_ROUTER.route(state["question"]); intent="legal_retrieval" if route.confidence>0 else "legal_consultation"
    return {"domain":route.domain,"intent":intent,"intent_confidence":route.confidence,"route_candidates":route.candidates,"trace":_trace(state,"intent_analysis",method=route.method,domain=route.domain)}
def task_planning(state:AgentState)->AgentState:
    plan=["route_to_domain_tool","retrieve_legal_basis","generate_answer","verify_answer"]
    return {"plan":plan,"current_step":0,"retry_count":state.get("retry_count",0),"trace":_trace(state,"task_planning",steps=plan)}
def tool_decision(state:AgentState,decision_generator:Callable[[str],dict[str,Any]|None]|None=None)->AgentState:
    decision=decision_generator(state["question"]) if decision_generator else None
    if decision:
        tool=decision.get("tool") or {}; domain=str(decision.get("domain") or state.get("domain","civil")); name=str(tool.get("name") or "")
        if name in {"search_labor_law","search_civil_law"} and name.endswith(f"{domain}_law"):
            args=tool.get("arguments") or {}; args={"query":str(args.get("query") or state["question"]),"limit":int(args.get("limit",5))}
            return {"domain":domain,"intent":decision.get("intent",state.get("intent")),"tool_name":name,"tool_arguments":args,"trace":_trace(state,"tool_decision",tool=name,method="model")}
    domain=state.get("domain","civil"); name=f"search_{domain}_law"
    return {"tool_name":name,"tool_arguments":{"query":state["question"],"limit":5},"trace":_trace(state,"tool_decision",tool=name,method="fallback")}
def tool_execution(state:AgentState,executor=None)->AgentState:
    executor=executor or DirectToolExecutor()
    try:return {"tool_result":executor.execute(state["tool_name"],state.get("tool_arguments",{})),"error":None,"trace":_trace(state,"tool_execution",tool=state["tool_name"],status="success")}
    except Exception as exc:return {"tool_result":"","error":str(exc),"trace":_trace(state,"tool_execution",tool=state["tool_name"],status="error",error=str(exc))}
def retrieval(state:AgentState)->AgentState:
    citations=[]
    for block in state.get("tool_result","").split("\n\n"):
        if block.startswith("[") and "]\n" in block:
            header,content=block.split("]\n",1); citations.append({"reference":header[1:],"content":content})
    return {"retrieved_documents":citations,"citations":citations,"trace":_trace(state,"retrieval",documents=len(citations))}
def generation(state:AgentState,answer_generator:Callable[[str,str,list[dict[str,Any]]],str]|None=None)->AgentState:
    q=state["question"]; evidence=state.get("tool_result",""); citations=state.get("citations",[])
    answer=answer_generator(q,evidence,citations) if answer_generator else ("根据检索到的法律依据，建议结合以下条款进一步判断：\n\n"+evidence if evidence else "暂未检索到足够的法律依据，无法给出可靠结论。")
    return {"answer":answer,"trace":_trace(state,"generation",answer_length=len(answer))}
def verification(state:AgentState)->AgentState:
    answer=state.get("answer",""); citations=state.get("citations",[]); ok=bool(answer.strip()) and bool(citations); v={"passed":ok,"citation_count":len(citations),"reason":"answer_and_citations_present" if ok else "missing_answer_or_citations"}
    return {"verification":v,"trace":_trace(state,"verification",**v)}
def route_after_tool(state:AgentState)->str:return "retry_or_end" if state.get("error") else "retrieval"
def route_after_verification(state:AgentState)->str:return "end" if state.get("verification",{}).get("passed") else ("retry" if state.get("retry_count",0)<1 else "end")
def retry_plan(state:AgentState)->AgentState:return {"retry_count":state.get("retry_count",0)+1,"plan":["retry_retrieval","generate_answer","verify_answer"],"tool_arguments":{"query":state["question"],"limit":8},"trace":_trace(state,"retry_plan",retry_count=state.get("retry_count",0)+1)}
