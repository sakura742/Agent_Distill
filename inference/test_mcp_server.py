# -*- coding: utf-8 -*-
"""
test_mcp_server.py

独立诊断工具：专门用来针对性验证和压测本地 legal_rag/server.py 是否能正常通过 MCP 管道通信。
"""

import os
import sys
import io
import asyncio
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 锁死编码，防止 Windows 终端中文打印崩溃
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 自动定位你的 server.py 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "legal_rag", "server.py"))

async def diagnose_mcp():
    print("===================================================", flush=True)
    print("🔍 [MCP 核心诊断] 正在启动本地 MCP 服务端连接测试...", flush=True)
    print(f"📂 目标服务器路径: {SERVER_PATH}", flush=True)
    print("===================================================", flush=True)

    if not os.path.exists(SERVER_PATH):
        print(f"❌ 致命错误: 未能在路径找到 server.py，请检查路径配置！", flush=True)
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

    print("⏳ 1. 正在尝试拉起子进程并建立 stdio 管道...", flush=True)
    try:
        async with stdio_client(server_params) as (read, write):
            print("➡️  [管道状态] stdio 管道建立成功！开始创建 Client 物理会话...", flush=True)
            
            async with ClientSession(read, write) as session:
                print("⏳ 2. 正在向 MCP 服务端发送 initialize 初始化握手信号...", flush=True)
                
                # 设置 10 秒超时，防止子进程由于内部 Embedding 加载慢卡死全局
                await asyncio.wait_for(session.initialize(), timeout=10.0)
                print("✅ [握手成功] MCP 服务端成功响应初始化，握手达成！", flush=True)

                print("\n⏳ 3. 正在获取服务器注册的所有可用工具列表...", flush=True)
                tools_list = await session.list_tools()
                print(f"🤖 [发现工具] 服务器当前挂载了以下工具:")
                for t in tools_list.tools:
                    print(f"   🛠️  工具名: {t.name} (描述: {t.description})", flush=True)

                # 4. 寻找一个合法的工具进行真实业务调用测试
                if not tools_list.tools:
                    print("⚠️  警告: 服务端虽然连接成功，但是没有注册任何工具！", flush=True)
                    return
                
                target_tool = tools_list.tools[0].name
                test_query = "工资"
                print(f"\n⏳ 4. 正在对目标工具 [{target_tool}] 发起真实检索测试，查询词: '{test_query}'...", flush=True)
                
                response = await asyncio.wait_for(
                    session.call_tool(target_tool, arguments={"query": test_query}),
                    timeout=15.0
                )
                
                print("\n🎉 ===================================================", flush=True)
                print("✅ 【测试成功】MCP 服务端完美通过闭环验证！", flush=True)
                print("===================================================", flush=True)
                if response.content:
                    print(f"📄 [样本检索结果]:\n{response.content[0].text[:300]}...", flush=True)
                else:
                    print("📄 [样本检索结果]: 成功返回，但内容为空。", flush=True)

    except asyncio.TimeoutError:
        print("\n❌ [诊断失败] 超时错误：MCP 服务端拉起成功，但在规定时间内未响应（初始化或检索超时）。", flush=True)
        print("💡 提示：这通常代表你的 server.py 内部仍在尝试请求网络，或者加载本地 Embedding 模型/数据库的速度极慢，请检查 server.py 是否真正全离线。", flush=True)
    except Exception as e:
        print("\n❌ [诊断失败] 管道通讯链路崩溃，捕获到底层异常：", flush=True)
        print("-" * 50, flush=True)
        traceback.print_exc()
        print("-" * 50, flush=True)
        print("💡 提示：请重点观察上方 Traceback 的最后几行，它会暴露究竟是哪个具体模块阻断了进程。", flush=True)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(diagnose_mcp())