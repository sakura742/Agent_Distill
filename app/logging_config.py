# -*- coding: utf-8 -*-
"""
app/logging_config.py
======================

Phase 1 重构新增：统一 logging 配置。

重构前全项目用 ``print(..., flush=True)`` 输出运行状态（inference_core.py、
agent_pipeline.py、gen_data.py、web/app.py 等），没有时间戳、没有级别、
无法按模块过滤、也无法在生产环境接入日志采集。

这里提供 ``get_logger(name)``，内部保证 ``logging.basicConfig`` 只被配置一次。
业务代码里原来的 ``print(...)`` 语句被替换为 ``logger.info(...)`` /
``logger.warning(...)`` / ``logger.error(...)``，**打印的内容和时机不变**，
只是换了输出通道，不影响任何业务逻辑或训练/推理结果。
"""

from __future__ import annotations

import logging
import sys

from configs.settings import settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """配置全局 logging，多次调用只生效一次（幂等）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取一个已完成全局配置的 logger。各模块用 ``get_logger(__name__)`` 调用。"""
    setup_logging()
    return logging.getLogger(name)
