"""
法律咨询评估平台 — FastAPI 后端
调用 inference_core.py 的两个模型，提供双列对比 + 多轮对话。
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.inference_core import load_models, run_inference

# ── FastAPI 应用 ──────────────────────────────────────────────
app = FastAPI(title="法律咨询评估平台")

# ── 启动时加载一次 ────────────────────────────────────────────
print("【web/app】加载模型和向量库...", flush=True)
resources = load_models()
print("【web/app】就绪 ✅", flush=True)

# ── 会话管理（内存） ──────────────────────────────────────────
sessions: dict[str, list] = {}  # session_id -> history list

MAX_HISTORY_TURNS = 3  # 保留最近 3 轮对话


def _trim_history(history: list) -> list:
    """保留最近 MAX_HISTORY_TURNS 轮（每轮 user+assistant 两条）"""
    return history[-(MAX_HISTORY_TURNS * 2):]


def _append_history(session_id: str, role: str, content: str):
    sessions[session_id].append({"role": role, "content": content})
    sessions[session_id] = _trim_history(sessions[session_id])


# ── 请求/响应模型 ────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    raw_answer: str
    tuned_answer: str
    reasoning: dict


class SessionResponse(BaseModel):
    session_id: str


# ── API 路由 ──────────────────────────────────────────────────
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

    history = sessions[req.session_id]
    result = run_inference(resources, user_query=req.message, history=history)

    # 将本轮对话追加到会话历史
    _append_history(req.session_id, "user", req.message)
    _append_history(req.session_id, "assistant", result["tuned_answer"])

    return ChatResponse(
        raw_answer=result["raw_answer"],
        tuned_answer=result["tuned_answer"],
        reasoning=result["reasoning"],
    )


# ── 静态文件 / 前端 ───────────────────────────────────────────
@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path)
