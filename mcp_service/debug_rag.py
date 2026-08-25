"""MCP 客户端连通性测试（原 legal_rag/debug_rag.py，迁移至 mcp_service/）。

原脚本用相对路径 ``args=["server.py"]`` 拉起子进程，隐含假设"当前工作目录就是
legal_rag/"——只要不在该目录下运行就会失败。Phase 1 顺手把它改成基于 __file__
的绝对路径（settings.mcp_server_path），使其可以从项目任意目录运行，属于修复
"可运行性"，不改变诊断逻辑本身。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from configs.settings import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


async def main():
    # 1. 告诉测试脚本，怎么直接用标准输入输出拉起你的法律知识库
    server_params = StdioServerParameters(
        command="python",
        args=[str(settings.mcp_server_path)],
        env=None
    )

    logger.info("正在初始化本地 MCP 法律知识库...")

    try:
        # 2. 建立本地管道连接
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化握手
                await session.initialize()
                logger.info("法律知识库连接成功！")

                # 查看有哪些可用工具
                tools = await session.list_tools()
                logger.info("监测到本地已注册的工具: %s", [t.name for t in tools.tools])

                # 3. 核心测试：直接模拟大模型发起调用
                logger.info("[测试开始] 正在检索法律私有知识库...")
                response = await session.call_tool(
                    "search_labor_law",
                    arguments={"query": "定期劳动合同什么时候转为不定期"}
                )

                # 4. 打印 RAG 捞出来的 PDF 原文
                logger.info("检索到的法律法条原文：\n%s", response.content[0].text)

    except Exception as e:
        logger.error("[测试发生错误] 正在抓取底层核心原因: %s", e)
        import traceback
        # 打印完整的错误堆栈
        traceback.print_exc()

        # 如果是 BaseExceptionGroup，把里面所有的子异常都打印出来
        if hasattr(e, 'exceptions'):
            logger.error("发现隐藏的子异常:")
            for i, sub_e in enumerate(e.exceptions):
                logger.error("  子异常 %d: %s", i + 1, sub_e)
                if hasattr(sub_e, '__context__') and sub_e.__context__:
                    logger.error("    上下文原因: %s", sub_e.__context__)


if __name__ == "__main__":
    # 修复 Windows 异步组件的经典 Bug
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
