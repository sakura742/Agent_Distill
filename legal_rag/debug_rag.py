import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 1. 告诉测试脚本，怎么直接用标准输入输出拉起你的法律知识库
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"], # 确保你的 server.py 就在旁边
        env=None
    )
    
    print("⏳ 正在初始化本地 MCP 法律知识库...")
    
    try:
        # 2. 建立本地管道连接
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化握手
                await session.initialize()
                print("✅ 法律知识库连接成功！")
                
                # 查看有哪些可用工具
                tools = await session.list_tools()
                print(f"📦 监测到本地已注册的工具: {[t.name for t in tools.tools]}")
                
                # 3. 核心测试：直接模拟大模型发起调用
                print("\n🚀 [测试开始] 正在检索法律私有知识库...")
                response = await session.call_tool(
                    "search_labor_law", 
                    arguments={"query": "定期劳动合同什么时候转为不定期"}
                )
                
                # 4. 打印 RAG 捞出来的 PDF 原文
                print("\n================= ⚖️ 检索到的法律法条原文 =================\n")
                print(response.content[0].text)
                print("\n=========================================================")

    except Exception as e:
        print("\n❌ [测试发生错误] 正在抓取底层核心原因...")
        import traceback
        # 打印完整的错误堆栈
        traceback.print_exc()
        
        # 如果是 BaseExceptionGroup，把里面所有的子异常都打印出来
        if hasattr(e, 'exceptions'):
            print("\n🔍 发现隐藏的子异常:")
            for i, sub_e in enumerate(e.exceptions):
                print(f"  子异常 {i+1}: {sub_e}")
                if hasattr(sub_e, '__context__') and sub_e.__context__:
                    print(f"    上下文原因: {sub_e.__context__}")

if __name__ == "__main__":
    # 修复 Windows 异步组件的经典 Bug
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())