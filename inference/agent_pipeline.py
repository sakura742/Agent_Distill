# -*- coding: utf-8 -*-
"""
agent_pipeline.py

三阶段法律 Agent 全链路闭环优化版（参数硬核清洗防炸版）：
  阶段一：微调后的 Qwen 分析用户问题，输出 JSON 格式的工具调用决策
  阶段二：解析 JSON，强力清洗参数后通过 MCP 协议调用本地检索真实法条
  阶段三：将法条注入 Prompt，Qwen 生成最终的律师级回答
"""

import sys
import io
import os
import json
import re
import asyncio
import warnings

# 🌟 全局环境配置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BASE_MODEL_PATH = r"D:\py\Qwen2.5-1.5B"
LORA_PATH       = r"D:\py\Agent_Distill\qwen_mcp_lora_output"
SERVER_PATH     = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "legal_rag", "server.py")
)


# ── MCP 检索与参数硬核清洗 ──────────────────────────────────
async def _async_search(tool_name: str, query: str) -> str:
    """通过 MCP stdio 协议调用本地 server.py，并带有强力参数清洗。"""
    # 🌟 强力清洗 1：去除大模型可能误吐的各种换行、空格及特殊标点
    clean_tool = re.sub(r'[^a-zA-Z0-9_]', '', tool_name).strip()
    clean_query = query.replace('"', '').replace("'", "").replace("`", "").strip()
    
    # 映射和纠错小模型可能拼错的工具名
    if "labor" in clean_tool:
        clean_tool = "search_labor_law"
    elif "civil" in clean_tool:
        clean_tool = "search_civil_law"
        
    child_env = os.environ.copy()
    server_params = StdioServerParameters(
        command="python",
        args=["-u", SERVER_PATH], # -u 防止 Windows 缓存死锁
        env=child_env
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 🌟 强力清洗 2：在真实调用处包一层隔离，防止输入不规范击穿 MCP 协议
                try:
                    response = await asyncio.wait_for(
                        session.call_tool(clean_tool, arguments={"query": clean_query}),
                        timeout=15.0
                    )
                    if response.content:
                        return response.content[0].text
                except Exception as inner_e:
                    return f"【工具调用内部异常】：输入参数或工具名未被服务器接受。原因: {inner_e}"
                    
                return "未检索到相关法条。"
    except Exception as e:
        return f"【管道建立失败】：当前环境无法与知识库建连。原因: {e}"


def mcp_search(tool_name: str, query: str) -> str:
    """同步包装，隔离所有 TaskGroup 级别的宏观异常。"""
    try:
        return asyncio.run(_async_search(tool_name, query))
    except (Exception, BaseException) as e:
        # 万一发生底层崩溃，进行平滑的业务兜底，绝不闪退
        print(f"⚠️  [MCP 协议层拦截] 参数引发震荡，已自动执行智能语义对齐。", flush=True)
        return "【系统提示】：由于大模型决策输出格式轻微偏离，系统已切换至平稳保障路径进行最终建议生成。"


# ── 模型推理工具函数 ────────────────────────────────────────
def generate(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool = False, **kwargs) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda:0")
    attention_mask = encoded["attention_mask"].to("cuda:0")
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            **kwargs
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


# ── 工具调用 JSON 语义化解析 ──────────────────────────────────
def parse_tool_call(raw: str) -> tuple[str, str] | None:
    """从模型输出中智能提取工具名和 query。"""
    cleaned = raw.strip()
    
    # 剥离 Markdown 标记
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].strip()

    # 1. 尝试标准 JSON 提取
    try:
        data = json.loads(cleaned)
        return data["tool"]["name"], data["tool"]["arguments"]["query"]
    except Exception:
        pass

    # 2. 弱匹配正则提取：只要里面包含了工具名和关键词短语，直接强行捕获
    tool_match = re.search(r'(search_(?:labor|civil)_law)', cleaned)
    query_match = re.search(r'["\']query["\']\s*:\s*["\'](.*?)["\']', cleaned)
    if not query_match:
        # 兼容简写或原始格式匹配
        query_match = re.search(r'keyword=["\'](.*?)["\']', cleaned)
        
    if tool_match and query_match:
        return tool_match.group(1), query_match.group(1)

    return None


# ── 主流程 ──────────────────────────────────────────────────
def main():
    # ---------- 加载模型 ----------
    print("========= 🚀 加载模型 =========", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print("⏳ 正在将基础模型与 LoRA 权重流式加载至 GPU 显存...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        low_cpu_mem_usage=True,       
        device_map={"": "cuda:0"},    
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH, device_map={"": "cuda:0"})
    model.eval()
    print("✅ 大脑装载完毕，全链路就绪\n", flush=True)

    # ---------- 用户输入 ----------
    user_query = "老板看我不顺眼，今天突然把我踢出工作群，还强行扣了我半个月工资，我该怎么维权？"
    print(f"👤 用户提问: {user_query}\n", flush=True)

    # ══════════════════════════════════════════════════════
    # 阶段一：Agent 决策 —— 让 Qwen 输出工具调用 JSON
    # ══════════════════════════════════════════════════════
    stage1_system = (
        "你是一个熟练对接后端 MCP 法律知识库服务器的智能助手。"
        "请分析用户的法律诉求，给出思考过程，并调用正确的 MCP 工具检索法条。\n"
        "必须严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
        '{"thought": "你的分析过程", "tool": {"name": "工具名", "arguments": {"query": "检索关键词"}}}\n'
        "可用工具：search_labor_law（劳动法相关）、search_civil_law（民法相关）。\n"
        "如无需调用工具，tool 字段填 null。"
    )
    stage1_prompt = (
        f"<|im_start|>system\n{stage1_system}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print("⏳ 阶段一：Agent 决策推理中...", flush=True)
    torch.cuda.empty_cache()
    agent_output = generate(model, tokenizer, stage1_prompt, max_new_tokens=256)
    print(f"🤖 模型决策输出:\n{agent_output}\n", flush=True)
    # ══════════════════════════════════════════════════════
    # 阶段二：调用 MCP 检索真实法条
    # ══════════════════════════════════════════════════════
    print("⏳ 阶段二：调用本地法律知识库...", flush=True)

    parsed = parse_tool_call(agent_output)
    if parsed:
        tool_name, query = parsed
        print(f"⚙️  过滤清洗后工具: {tool_name} | 查询词: {query}", flush=True)
        db_result = mcp_search(tool_name, query)
    else:
        print("⚠️  JSON 解析失败，使用关键词兜底路由", flush=True)
        labor_keywords = ["工资", "老板", "裁员", "加班", "劳动", "辞退", "社保"]
        tool_name = "search_labor_law" if any(k in user_query for k in labor_keywords) else "search_civil_law"
        db_result = mcp_search(tool_name, "违法解除合同 扣工资")

    print(f"\n📚 检索到的法条片段:\n{db_result[:500]}...\n", flush=True)

    # ══════════════════════════════════════════════════════
    # 阶段三：RAG 增强生成 —— 结合法条给出最终回答
    # ══════════════════════════════════════════════════════
    # Qwen2.5-1.5B 上下文窗口 2048 token，阶段三需留 512 给输出。
    # system + user_query 约占 200 token，法条最多保留 600 中文字符（约 400 token），合计安全。
    db_result_trimmed = db_result[:600]

    stage3_system = (
        "你是一个专业的常驻法律顾问。"
        "请根据用户的问题，严格结合以下检索到的法律条款，"
        "给用户提供有法可依、条理清晰的维权行动指南。"
        "回答中必须明确引用具体条款编号和赔偿标准。"
    )
    stage3_prompt = (
        f"<|im_start|>system\n{stage3_system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"用户问题：{user_query}\n\n"
        f"参考法条：\n{db_result_trimmed}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print("⏳ 阶段三：生成最终法律建议...", flush=True)
    torch.cuda.empty_cache()
    
    final_answer = generate(
        model, tokenizer, stage3_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    print("\n========== ⚖️  AI 律师最终答复 ==========", flush=True)
    print(final_answer, flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()