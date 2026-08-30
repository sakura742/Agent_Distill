# -*- coding: utf-8 -*-
"""
Phase 1 新增：不依赖 torch / transformers / chromadb / langchain / mcp / openai
等重量级三方库的"结构性"测试。

这些库需要 GPU/模型权重/网络环境，不在本仓库的测试沙箱里安装，因此这里只验证：
  1. 新增的配置系统（configs.settings）能正确加载、类型正确、默认值与重构前的
     硬编码值一致；
  2. 统一异常体系（app.exceptions）可用；
  3. 统一 logging（app.logging_config）可用；
  4. 九个新模块目录都有 __init__.py，是合法的 Python 包。

需要真实模型/向量库跑通的训练、推理、检索流程，见
docs/phase1_refactor_notes.md 中"如何运行原来的功能"一节人工验证步骤。
"""

import importlib
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Settings 会在模块导入时加载 .env。默认值测试必须显式清除配置环境变量，
# 否则开发机上的 .env / shell 环境会污染“默认值”测试。
_SETTINGS_ENV_VARS = (
    "AGENT_DISTILL_BASE_MODEL_PATH",
    "AGENT_DISTILL_LORA_OUTPUT_DIR",
    "AGENT_DISTILL_MERGED_MODEL_DIR",
    "AGENT_DISTILL_DATA_DIR",
    "AGENT_DISTILL_CHROMA_DB_DIR",
    "AGENT_DISTILL_TRAIN_DATA_PATH",
    "AGENT_DISTILL_TOOLS_CONFIG_PATH",
    "AGENT_DISTILL_MCP_SERVER_PATH",
    "AGENT_DISTILL_EMBEDDING_MODEL",
    "AGENT_DISTILL_RERANKER_MODEL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "AGENT_DISTILL_MAX_HISTORY_TURNS",
    "AGENT_DISTILL_LAW_SNIPPET_LIMIT",
    "AGENT_DISTILL_RETRIEVAL_TOP_K",
    "AGENT_DISTILL_LOG_LEVEL",
)


def test_settings_loads_with_defaults(monkeypatch):
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    from configs.settings import Settings

    # 默认值应与重构前硬编码值完全一致（不设置任何环境变量的情况下）
    s = Settings()
    assert s.base_model_path == r"D:\py\Qwen2.5-1.5B"
    assert str(s.lora_output_dir) == r"D:\py\Agent_Distill\qwen_mcp_lora_output"
    assert s.embedding_model_name == "shibing624/text2vec-base-chinese"
    assert s.max_history_turns == 3
    assert s.law_snippet_limit == 600
    assert s.deepseek_base_url == "https://api.deepseek.com"
    assert s.deepseek_model == "deepseek-chat"


def test_settings_paths_are_path_objects(monkeypatch):
    # Never inspect the module-level singleton here: it may have been created
    # during test collection while the developer shell had Phase 6 overrides
    # such as AGENT_DISTILL_CHROMA_DB_DIR configured.
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    from configs.settings import Settings

    s = Settings()
    assert str(s.data_dir).endswith("data")
    assert str(s.chroma_db_dir).endswith(os.path.join("knowledge", "chroma_db"))
    assert str(s.train_data_path).endswith(
        os.path.join("distill", "data", "agent_distill_train.jsonl")
    )
    assert str(s.tools_config_path).endswith(
        os.path.join("distill", "tools_config.json")
    )


def test_settings_env_override(monkeypatch):
    custom = os.path.join(PROJECT_ROOT, "knowledge", "custom_chroma")
    monkeypatch.setenv("AGENT_DISTILL_CHROMA_DB_DIR", custom)
    from configs.settings import Settings

    s = Settings()
    assert str(s.chroma_db_dir).endswith(os.path.join("knowledge", "custom_chroma"))


@pytest.mark.parametrize(
    "module_name",
    [
        "app.exceptions",
        "app.logging_config",
        "configs.settings",
        "knowledge.legal_schema",
        "knowledge.domain_config",
        "knowledge.legal_chunker",
        "knowledge.topic_enrichment",
        "knowledge.legal_concepts",
        "knowledge.query_rewrite",
        "evaluation.retrieval_benchmark",
        "evaluation.retrieval_diagnostics",
        "evaluation.reranker_diagnostics",
        "evaluation.router_benchmark",
        "distill.filter_trajectory",
        "distill.prepare_phase5_data",
        "distill.audit_phase5_data",
        "distill.train_phase5",
    ],
)
def test_business_modules_import_or_skip_missing_heavy_deps(module_name):
    heavy = {"torch", "transformers", "chromadb", "langchain", "langchain_chroma", "sentence_transformers"}
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        if any(dep in str(exc) for dep in heavy):
            pytest.skip(f"缺少重量级依赖: {exc}")
        raise
    except Exception as exc:
        pytest.fail(f"{module_name} import 失败，且不是缺依赖导致的: {exc!r}")
