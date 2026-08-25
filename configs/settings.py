# -*- coding: utf-8 -*-
"""
configs/settings.py
====================

Phase 1 重构新增：统一配置系统。

在重构之前，模型路径、数据路径、向量库路径、DeepSeek API Key 等全部以字符串
字面量的形式硬编码在 8+ 个不同文件里（多为 Windows 绝对路径 ``D:\\py\\...``），
且 API Key 直接提交进了 Git 历史。

本模块把它们统一收敛为一个 ``Settings`` 数据类，所有取值都 **优先读环境变量**，
**读不到环境变量时才回退到与重构前完全相同的默认值** —— 这样即使不配置任何
环境变量，行为也和重构前一致（保持项目在没有 .env 的情况下依然"能跑"）。

唯一的例外是 ``deepseek_api_key``：重构前它被硬编码在 ``distill/gen_data.py``
第 7 行，属于已经泄露的安全问题；重构后**没有默认值**，必须显式通过环境变量
``DEEPSEEK_API_KEY`` 提供，读不到时相关代码会抛出
``app.exceptions.ConfigurationError``，而不是静默使用一个写死的密钥。

用法：
    from configs.settings import settings
    print(settings.base_model_path)

覆盖方式（任选其一）：
    1. 直接 export 环境变量：
       export AGENT_DISTILL_BASE_MODEL_PATH=/models/Qwen2.5-1.5B
    2. 在项目根目录放一个 ``.env`` 文件（如果安装了 python-dotenv 会被自动加载；
       未安装也不影响其余功能，只是不会自动读取 .env，需要手动 export）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 项目根目录：configs/settings.py -> configs/ -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 尝试加载 .env（可选依赖，没装也不报错，只是跳过）
try:  # pragma: no cover - 纯 IO 便利性代码
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # ── 项目路径 ──────────────────────────────────────────────
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(
        default_factory=lambda: _env_path("AGENT_DISTILL_DATA_DIR", PROJECT_ROOT / "data")
    )
    chroma_db_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_DISTILL_CHROMA_DB_DIR", PROJECT_ROOT / "knowledge" / "chroma_db"
        )
    )
    train_data_path: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_DISTILL_TRAIN_DATA_PATH",
            PROJECT_ROOT / "distill" / "data" / "agent_distill_train.jsonl",
        )
    )
    tools_config_path: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_DISTILL_TOOLS_CONFIG_PATH", PROJECT_ROOT / "distill" / "tools_config.json"
        )
    )
    # 注意：重构前 train.py / inference_core.py / agent_pipeline.py / evaluate.py /
    # merge_model.py 五个文件里硬编码的 LoRA 目录字面量完全相同
    # (r"D:\py\Agent_Distill\qwen_mcp_lora_output")。为了不引入"训练产出路径"和
    # "推理读取路径"默认值不一致的新 bug，这里只保留一个 lora_output_dir 字段，
    # 默认值取与原来完全相同的字符串，训练/推理/合并/评估四处全部读同一个值。
    lora_output_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_DISTILL_LORA_OUTPUT_DIR", Path(r"D:\py\Agent_Distill\qwen_mcp_lora_output")
        )
    )
    merged_model_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_DISTILL_MERGED_MODEL_DIR", PROJECT_ROOT / "qwen_merged"
        )
    )
    mcp_server_path: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_DISTILL_MCP_SERVER_PATH", PROJECT_ROOT / "mcp_service" / "server.py"
        )
    )

    # ── 模型 ──────────────────────────────────────────────────
    # 默认值与重构前 inference_core.py / train.py 中硬编码的值完全一致，
    # 保证不配置环境变量时行为不变（仍是原来那个 Windows 路径，需用户自行配置）。
    base_model_path: str = field(
        default_factory=lambda: _env("AGENT_DISTILL_BASE_MODEL_PATH", r"D:\py\Qwen2.5-1.5B")
    )
    embedding_model_name: str = field(
        default_factory=lambda: _env(
            "AGENT_DISTILL_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
        )
    )

    # ── DeepSeek 教师模型 API（安全修复：不再有硬编码默认值）──────
    deepseek_api_key: Optional[str] = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(
        default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    deepseek_model: str = field(default_factory=lambda: _env("DEEPSEEK_MODEL", "deepseek-chat"))

    # ── 推理行为参数（与重构前 inference_core.py 中的常量默认值一致）──
    max_history_turns: int = field(
        default_factory=lambda: _env_int("AGENT_DISTILL_MAX_HISTORY_TURNS", 3)
    )
    law_snippet_limit: int = field(
        default_factory=lambda: _env_int("AGENT_DISTILL_LAW_SNIPPET_LIMIT", 600)
    )
    retrieval_top_k: int = field(
        default_factory=lambda: _env_int("AGENT_DISTILL_RETRIEVAL_TOP_K", 3)
    )

    # ── 日志 ──────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: _env("AGENT_DISTILL_LOG_LEVEL", "INFO"))


# 模块级单例，其余代码统一 `from configs.settings import settings`
settings = Settings()
