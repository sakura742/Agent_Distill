"""法律咨询评估平台 — FastAPI 后端。"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import settings
from app.logging_config import get_logger

logger = get_logger(__name__)
app = FastAPI(title="法律咨询评估平台")

# 模型/向量库在首次实际请求时加载，避免 `import web.app` 触发本机模型加载。
_resources = None
sessions: dict[str, list] = {}
MAX_HISTORY_TURNS = settings.max_history_turns


def _get_resources():
    global _resources
    if _resources is None:
        from agent.inference_core import load_models
        logger.info("加载模型和向量库...")
        _resources = load_models()
        logger.info("就绪")
    return _resources


def _trim_history(history: list) -> list:
    return history[-(MAX_HISTORY_TURNS * 2):]


def _append_history(session_id: str, role: str, content: str):
    sessions[session_id].append({"role": role, "content": content})
    sessions[session_id] = _trim_history(sessions[session_id])


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    raw_answer: str
    tuned_answer: str
    reasoning: dict


class SessionResponse(BaseModel):
    session_id: str


@app.post("/session/new", response_model=SessionResponse)
def new_session():
    session_id = uuid.uuid4().hex[:12]
    sessions[session_id] = []
    return {"session_id": session_id}


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "history": sessions[session_id]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在，请先创建会话")
    from agent.inference_core import run_inference
    result = run_inference(_get_resources(), user_query=req.message, history=sessions[req.session_id])
    _append_history(req.session_id, "user", req.message)
    _append_history(req.session_id, "assistant", result["tuned_answer"])
    return ChatResponse(
        raw_answer=result["raw_answer"],
        tuned_answer=result["tuned_answer"],
        reasoning=result["reasoning"],
    )


@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path)
