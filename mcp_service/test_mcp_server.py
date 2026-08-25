# -*- coding: utf-8 -*-
"""
mcp_service/test_mcp_server.py

独立诊断工具：专门用来针对性验证和压测 mcp_service/server.py 是否能正常通过
MCP 管道通信。（原 inference/test_mcp_server.py，迁移至 mcp_service/，与
debug_rag.py 功能高度重叠 —— 按 Phase 1"不删除现有功能"原则两者都保留，
去重留给后续阶段。）
"""

import os
import sys
import asyncio
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from configs.settings import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

SERVER_PATH = str(settings.mcp_server_path)


async def diagnose_mcp():
    logger.info("[MCP 核心诊断] 正在启动本地 MCP 服务端连接测试...")
    logger.info("目标服务器路径: %s", SERVER_PATH)

    if not os.path.exists(SERVER_PATH):
        logger.error("致命错误: 未能在路径找到 server.py，请检查路径配置！")
        return

    # 1. 完整克隆当前虚拟环境的环境变量，确保子进程能找到 Python 依赖和离线配置
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["HF_HUB_OFFLINE"] = "1"  # 锁定离线

    # 2. 明确拉起参数（使用 -u 强制 Python 刷新标准输出流，防止 Windows 管道死锁）
    server_params = StdioServerParameters(
        command="python",
        args=["-u", SERVER_PATH],
        env=child_env
    )

    logger.info("1. 正在尝试拉起子进程并建立 stdio 管道...")
    try:
        async with stdio_client(server_params) as (read, write):
            logger.info("[管道状态] stdio 管道建立成功！开始创建 Client 物理会话...")

            async with ClientSession(read, write) as session:
                logger.info("2. 正在向 MCP 服务端发送 initialize 初始化握手信号...")

                # 设置 10 秒超时，防止子进程由于内部 Embedding 加载慢卡死全局
                await asyncio.wait_for(session.initialize(), timeout=10.0)
                logger.info("[握手成功] MCP 服务端成功响应初始化，握手达成！")

                logger.info("3. 正在获取服务器注册的所有可用工具列表...")
                tools_list = await session.list_tools()
                logger.info("[发现工具] 服务器当前挂载了以下工具:")
                for t in tools_list.tools:
                    logger.info("   工具名: %s (描述: %s)", t.name, t.description)

                # 4. 寻找一个合法的工具进行真实业务调用测试
                if not tools_list.tools:
                    logger.warning("警告: 服务端虽然连接成功，但是没有注册任何工具！")
                    return

                target_tool = tools_list.tools[0].name
                test_query = "工资"
                logger.info("4. 正在对目标工具 [%s] 发起真实检索测试，查询词: '%s'...", target_tool, test_query)

                response = await asyncio.wait_for(
                    session.call_tool(target_tool, arguments={"query": test_query}),
                    timeout=15.0
                )

                logger.info("【测试成功】MCP 服务端完美通过闭环验证！")
                if response.content:
                    logger.info("[样本检索结果]:\n%s...", response.content[0].text[:300])
                else:
                    logger.info("[样本检索结果]: 成功返回，但内容为空。")

    except asyncio.TimeoutError:
        logger.error("[诊断失败] 超时错误：MCP 服务端拉起成功，但在规定时间内未响应（初始化或检索超时）。")
        logger.error("提示：这通常代表你的 server.py 内部仍在尝试请求网络，或者加载本地 Embedding 模型/数据库的速度极慢，请检查 server.py 是否真正全离线。")
    except Exception as e:
        logger.error("[诊断失败] 管道通讯链路崩溃，捕获到底层异常: %s", e)
        traceback.print_exc()
        logger.error("提示：请重点观察上方 Traceback 的最后几行，它会暴露究竟是哪个具体模块阻断了进程。")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(diagnose_mcp())
