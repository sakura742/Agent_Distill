"""FastAPI server exposing Qwen3.5 through an OpenAI-compatible subset."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .qwen35 import GenerationConfig, Qwen35Service

service = Qwen35Service()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield
    service.unload()


app = FastAPI(title="Agent Distill Qwen3.5 Serving", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return service.health()


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    text = service.generate(
        request.messages,
        GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        ),
    )
    return {
        "object": "chat.completion",
        "model": service.model_path,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }
