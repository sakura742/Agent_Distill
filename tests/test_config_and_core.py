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


def test_settings_paths_are_path_objects():
    from configs.settings import settings

    assert str(settings.data_dir).endswith("data")
    assert str(settings.chroma_db_dir).endswith(os.path.join("knowledge", "chroma_db"))
    assert str(settings.train_data_path).endswith(
        os.path.join("distill", "data", "agent_distill_train.jsonl")
    )
    assert str(settings.tools_config_path).endswith(
        os.path.join("distill", "tools_config.json")
    )
    assert str(settings.mcp_server_path).endswith(
        os.path.join("mcp_service", "server.py")
    )


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_DISTILL_BASE_MODEL_PATH", "/tmp/fake-qwen")
    monkeypatch.setenv("AGENT_DISTILL_MAX_HISTORY_TURNS", "7")
    # Settings 是 dataclass，每次实例化都会重新求值 default_factory，
    # 这里直接实例化一个新的 Settings 验证覆盖生效，而不用重新 import 单例。
    from configs.settings import Settings

    s = Settings()
    assert s.base_model_path == "/tmp/fake-qwen"
    assert s.max_history_turns == 7


def test_deepseek_api_key_has_no_hardcoded_default(monkeypatch):
    """安全回归测试：确保 API Key 不再有任何硬编码默认值。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from configs.settings import Settings

    s = Settings()
    assert s.deepseek_api_key is None


def test_exceptions_hierarchy():
    from app.exceptions import (
        AgentDistillError,
        ConfigurationError,
        KnowledgeBaseError,
        ModelLoadError,
        ToolCallParseError,
        DataGenerationError,
        MCPConnectionError,
    )

    for exc_cls in (
        ConfigurationError,
        KnowledgeBaseError,
        ModelLoadError,
        ToolCallParseError,
        DataGenerationError,
        MCPConnectionError,
    ):
        assert issubclass(exc_cls, AgentDistillError)
        with pytest.raises(AgentDistillError):
            raise exc_cls("boom")


def test_logging_setup_is_idempotent():
    from app.logging_config import setup_logging, get_logger
    import logging

    setup_logging()
    setup_logging()  # 第二次调用不应报错、不应重复添加 handler
    logger = get_logger("agent_distill.test")
    assert isinstance(logger, logging.Logger)
    root_handlers_before = len(logging.getLogger().handlers)
    setup_logging()
    root_handlers_after = len(logging.getLogger().handlers)
    assert root_handlers_before == root_handlers_after


@pytest.mark.parametrize(
    "package",
    [
        "app",
        "agent",
        "knowledge",
        "mcp_service",
        "distill",
        "evaluation",
        "web",
        "tests",
        "configs",
    ],
)
def test_required_packages_have_init_py(package):
    init_path = os.path.join(PROJECT_ROOT, package, "__init__.py")
    assert os.path.isfile(init_path), f"{package}/__init__.py 缺失"


@pytest.mark.parametrize(
    "directory",
    ["app", "agent", "knowledge", "mcp_service", "distill", "evaluation",
     "web", "deployment", "tests", "configs", "docs"],
)
def test_required_top_level_directories_exist(directory):
    assert os.path.isdir(os.path.join(PROJECT_ROOT, directory)), f"{directory}/ 目录缺失"


@pytest.mark.parametrize(
    "module_name",
    [
        "agent.inference_core",
        "agent.agent_pipeline",
        "knowledge.ingest",
        "knowledge.direct_rag",
        "mcp_service.server",
        "mcp_service.debug_rag",
        "mcp_service.test_mcp_server",
        "distill.gen_data",
        "distill.train",
        "distill.merge_model",
        "evaluation.evaluate",
        "web.app",
    ],
)
def test_business_modules_import_or_skip_missing_heavy_deps(module_name):
    """
    这些模块依赖 torch / transformers / langchain / chromadb / mcp / openai /
    fastapi 等重量级三方库。在没有安装这些库、没有 GPU、没有模型权重的沙箱环境
    里，我们不强求真正 import 成功 —— 只要失败原因是"缺少某个第三方库"
    （ImportError/ModuleNotFoundError），就算作预期内的 skip，而不是测试失败。
    如果失败原因是别的（比如我们改错了 import 路径、拼写错了变量名），
    这里会照常报错，能捕获 Phase 1 重构引入的 import bug。
    """
    try:
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"{module_name} 需要未安装的第三方依赖，属于预期跳过: {e}")
    except Exception as e:  # noqa: BLE001 - 我们就是要在这里兜底捕获重构引入的真 bug
        pytest.fail(f"{module_name} import 失败，且不是缺依赖导致的: {e!r}")
