# -*- coding: utf-8 -*-
"""统一项目配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_DATA_DIR", PROJECT_ROOT / "data"))
    chroma_db_dir: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_CHROMA_DB_DIR", PROJECT_ROOT / "knowledge" / "chroma_db"))
    train_data_path: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_TRAIN_DATA_PATH", PROJECT_ROOT / "distill" / "data" / "agent_distill_train.jsonl"))
    trajectory_data_path: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_TRAJECTORY_DATA_PATH", PROJECT_ROOT / "distill" / "data" / "agent_trajectory.jsonl"))
    hard_example_path: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_HARD_EXAMPLE_PATH", PROJECT_ROOT / "distill" / "data" / "hard_examples.jsonl"))
    tools_config_path: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_TOOLS_CONFIG_PATH", PROJECT_ROOT / "distill" / "tools_config.json"))
    lora_output_dir: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_LORA_OUTPUT_DIR", Path(r"D:\py\Agent_Distill\qwen_mcp_lora_output")))
    merged_model_dir: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_MERGED_MODEL_DIR", PROJECT_ROOT / "qwen_merged"))
    mcp_server_path: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_MCP_SERVER_PATH", PROJECT_ROOT / "mcp_service" / "server.py"))

    qwen35_model_path: str = field(default_factory=lambda: _env("AGENT_DISTILL_QWEN35_MODEL_PATH", r"D:\py\models"))
    qwen35_lora_output_dir: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_QWEN35_LORA_OUTPUT_DIR", PROJECT_ROOT / "qwen35_lora"))
    qwen35_decision_lora_output_dir: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_QWEN35_DECISION_LORA_OUTPUT_DIR", PROJECT_ROOT / "qwen35_decision_lora"))
    qwen35_answer_lora_output_dir: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_QWEN35_ANSWER_LORA_OUTPUT_DIR", PROJECT_ROOT / "qwen35_answer_lora"))
    phase5_decision_data_path: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_PHASE5_DECISION_DATA_PATH", PROJECT_ROOT / "distill" / "data" / "phase5_decision.jsonl"))
    phase5_answer_data_path: Path = field(default_factory=lambda: _env_path("AGENT_DISTILL_PHASE5_ANSWER_DATA_PATH", PROJECT_ROOT / "distill" / "data" / "phase5_answer.jsonl"))

    base_model_path: str = field(default_factory=lambda: _env("AGENT_DISTILL_BASE_MODEL_PATH", r"D:\py\Qwen2.5-1.5B"))
    embedding_model_name: str = field(default_factory=lambda: _env("AGENT_DISTILL_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"))
    reranker_model_name: Optional[str] = field(default_factory=lambda: _env("AGENT_DISTILL_RERANKER_MODEL"))
    retrieval_min_score: Optional[float] = field(default_factory=lambda: (_env_float("AGENT_DISTILL_RETRIEVAL_MIN_SCORE", 0.45) if os.environ.get("AGENT_DISTILL_RETRIEVAL_MIN_SCORE") else None))
    retrieval_candidate_multiplier: int = field(default_factory=lambda: _env_int("AGENT_DISTILL_RETRIEVAL_CANDIDATE_MULTIPLIER", 4))
    retrieval_rerank: bool = field(default_factory=lambda: _env("AGENT_DISTILL_RETRIEVAL_RERANK", "0") == "1")
    retrieval_query_rewrite: bool = field(default_factory=lambda: _env("AGENT_DISTILL_RETRIEVAL_QUERY_REWRITE", "0") == "1")
    deepseek_api_key: Optional[str] = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_model: str = field(default_factory=lambda: _env("DEEPSEEK_MODEL", "deepseek-chat"))
    max_history_turns: int = field(default_factory=lambda: _env_int("AGENT_DISTILL_MAX_HISTORY_TURNS", 3))
    law_snippet_limit: int = field(default_factory=lambda: _env_int("AGENT_DISTILL_LAW_SNIPPET_LIMIT", 600))
    retrieval_top_k: int = field(default_factory=lambda: _env_int("AGENT_DISTILL_RETRIEVAL_TOP_K", 3))
    trajectory_max_tokens: int = field(default_factory=lambda: _env_int("AGENT_DISTILL_TRAJECTORY_MAX_TOKENS", 1024))
    log_level: str = field(default_factory=lambda: _env("AGENT_DISTILL_LOG_LEVEL", "INFO"))
    hf_local_files_only: bool = field(default_factory=lambda: _env("HF_HUB_OFFLINE", "0") == "1")


settings = Settings()
