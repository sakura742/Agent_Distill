import json
import time
from openai import OpenAI

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key="sk-2ff40f7f13d04fe898cd92f190a14bce", # ⚠️ 记得换成你自己的 DeepSeek API Key
    base_url="https://api.deepseek.com"
)

# 载入工具库定义
with open("tools_config.json", "r", encoding="utf-8") as f:
    tools_schema = f.read()

# 针对法律 MCP 链路优化的 Master Prompt
SYSTEM_PROMPT = f"""你是一个顶级的大模型 Agent 训练数据生成专家。
你的任务是为轻量化法律大模型（Qwen 1.5B）的微调（LoRA）生成高质量的 MCP 工具调用（Tool Calling）和思维链（CoT）数据集。

# 业务背景
我们将微调一个 1.5B 的小模型，使其能够作为前端 Agent 完美对接后端的法律 MCP 服务器（装载了民法和劳动法）。

# 后端 MCP 服务器可用工具 (Tools)
<tools>
{tools_schema}
</tools>

# 任务要求
请模拟用户在日常生活中可能遇到的法律纠纷（涵盖民法、劳动法、寒暄等），生成 20 条场景完全不重复的训练样本。
样本类型必须严格包含以下三种（按比例分配）：
1. 劳动法咨询 (约 40%)：如加班费、降薪、裁员赔偿、未签劳动合同等（决定调用 search_labor_law）。
2. 民法典咨询 (约 40%)：如租房合同纠纷、离婚财产分割、邻里侵权、借钱不还等（决定调用 search_civil_law）。
3. 负样本 (约 20%)：日常闲聊或无法用法律条文直接检索的宏观闲扯（如“你觉得人工智能会取代律师吗？”、“你好”），Agent的thought应写明无需调用MCP工具，直接回复。

# 输出格式
必须直接返回标准的 JSON 数组，不要包含 markdown 标记（如 ```json）。格式如下：
[
  {{
    "user_query": "用户遇到的法律纠纷描述",
    "thought": "Agent的详细思维链：识别纠纷属于民法还是劳动法域 -> 拆解出检索关键词 -> 决定调用哪个 MCP 工具获取法条支撑。",
    "tool_call": {{ "name": "工具名", "arguments": {{ ... }} }} // 如果不需要调用工具，此字段填 null
  }}
]
"""

def generate_batch(batch_idx):
    user_prompt = f"请开始第 {batch_idx} 批次的生成。请提供 20 条全新的、符合上述三种法律场景分布的高质量数据。"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8, # 稍微拉高，让模拟的法律案例更丰富生动
            max_tokens=4000
        )
        raw_content = response.choices[0].message.content.strip()
        
        # 剥离 markdown 标签
        if raw_content.startswith("```json"):
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1].split("```")[0].strip()

        samples = json.loads(raw_content)
        
        # 写入训练集 (agent_distill_train.jsonl)
        with open("agent_distill_train.jsonl", "a", encoding="utf-8") as f:
            for sample in samples:
                if sample["tool_call"]:
                    action_str = f"思考：{sample['thought']}\n行动：调用法律 MCP 服务器工具 {sample['tool_call']['name']}，参数为 {json.dumps(sample['tool_call']['arguments'], ensure_ascii=False)}"
                else:
                    action_str = f"思考：{sample['thought']}\n行动：无需调用 MCP 法律检索。直接结合常识友好回复。"

                formatted_entry = {
                    "instruction": "你是一个熟练对接后端 MCP 法律知识库服务器的智能助手。请分析用户的法律诉求，给出你的思考过程，并在必要时调用正确的 MCP 工具检索法条。",
                    "input": sample["user_query"],
                    "output": action_str
                }
                f.write(json.dumps(formatted_entry, ensure_ascii=False) + "\n")
        print(f"【成功】第 {batch_idx} 批次成功，成功写入 {len(samples)} 条法律 Agent 蒸馏数据。")
    except Exception as e:
        print(f"【失败】第 {batch_idx} 批次异常: {e}")

if __name__ == "__main__":
    # 循环运行 10 次，获取 200 条核心法律蒸馏数据
    for i in range(1, 11):
        print(f"开始生成第 {i} 批法律 MCP 数据...")
        generate_batch(batch_idx=i)
        time.sleep(1)
    print("🎉 阶段二数据全量生成完毕！请检查 D:\\py\\Agent_Distill\\agent_distill_train.jsonl。")