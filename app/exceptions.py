# -*- coding: utf-8 -*-
"""
app/exceptions.py
==================

Phase 1 重构新增：统一异常体系。

重构前，各模块各自处理错误：有的直接让异常裸抛（``server.py`` 里
``raise FileNotFoundError(...)``），有的把异常吞掉转成字符串塞进返回值里
（``agent_pipeline.py`` 的 ``mcp_search`` 捕获 ``(Exception, BaseException)``
后返回一句提示文本），排查问题时很难区分"配置错了""模型加载失败"还是
"向量库没建"。

这里只新增一套异常**类型**，用于替换原来裸的 ``Exception`` / 字符串错误提示，
不改变任何业务判断逻辑本身（比如 agent_pipeline 原本"出错也不崩溃、返回兜底
文案"的产品行为予以保留，只是把"捕获到什么类型的错误"这件事变得可判别）。
"""

from __future__ import annotations


class AgentDistillError(Exception):
    """全部自定义异常的基类，方便上层统一 ``except AgentDistillError``。"""


class ConfigurationError(AgentDistillError):
    """必需的配置缺失或非法（例如未设置 DEEPSEEK_API_KEY）。"""


class KnowledgeBaseError(AgentDistillError):
    """向量库 / 知识库不可用：未构建、路径不存在、检索失败等。"""


class ModelLoadError(AgentDistillError):
    """基座模型或 LoRA 权重加载失败。"""


class ToolCallParseError(AgentDistillError):
    """模型输出的工具调用 JSON 无法被解析（JSON 与正则兜底都失败）。"""


class DataGenerationError(AgentDistillError):
    """调用教师模型（DeepSeek）生成蒸馏数据失败。"""


class MCPConnectionError(AgentDistillError):
    """MCP 客户端无法与 MCP Server 建立连接或完成工具调用。"""
