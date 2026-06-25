# -*- coding: utf-8 -*-
"""
evaluate.py

对比原始 Qwen2.5-1.5B 与微调后模型在法律 Agent 工具调用上的准确率。

评估方式（双层）：
  主评估：JSON 解析后对 tool.name 字段做精确匹配
  降级评估：若模型输出无法解析为 JSON，退回字符串包含匹配
每题记录命中方式，最终分别统计两种命中数，便于判断模型输出规范程度。
"""

import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── 工具描述（与 agent_pipeline 保持一致，只包含已实现的工具）──────────
TOOLS_DESC = (
    "可用工具：\n"
    "1. search_labor_law(query: str) -> 搜索劳动法及劳动合同法相关条文\n"
    "2. search_civil_law(query: str) -> 搜索民法典相关条文\n"
)

SYSTEM_PROMPT = (
    "你是一个熟练对接后端 MCP 法律知识库服务器的智能助手。\n"
    "请分析用户的法律诉求，给出思考过程，并调用正确的 MCP 工具检索法条。\n"
    "必须严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
    '{"thought": "你的分析过程", "tool": {"name": "工具名", "arguments": {"query": "检索关键词"}}}\n'
    f"{TOOLS_DESC}"
    "如无需调用工具，tool 字段填 null。"
)

# ── 测试集（5 题全部使用已实现的工具）────────────────────────────────
test_set = [
    {
        "id": 1,
        "tag": "劳动法-违法辞退",
        "question": "我在公司干了3年，上周突然被口头通知不用来了，有赔偿吗？",
        "expected_tool": "search_labor_law",
        "standard_cot": "用户被公司口头辞退，涉及违法解除劳动合同和经济补偿金，属于劳动法范畴。",
        "standard_call": '{"tool": {"name": "search_labor_law", "arguments": {"query": "违法解除劳动合同 经济补偿"}}}'
    },
    {
        "id": 2,
        "tag": "民法-借贷纠纷",
        "question": "我借给朋友5万块钱，他一直不还，我该怎么办？",
        "expected_tool": "search_civil_law",
        "standard_cot": "用户涉及民间借贷纠纷，属于民法债权债务关系。",
        "standard_call": '{"tool": {"name": "search_civil_law", "arguments": {"query": "民间借贷 债务追偿"}}}'
    },
    {
        "id": 3,
        "tag": "民法-租房押金",
        "question": "搬家后房东说我损坏了墙面要扣我全部押金，但那是正常磨损，我该怎么办？",
        "expected_tool": "search_civil_law",
        "standard_cot": "租房押金纠纷涉及租赁合同和正常损耗责任划分，属于民法典范畴。",
        "standard_call": '{"tool": {"name": "search_civil_law", "arguments": {"query": "租赁合同 押金 正常损耗"}}}'
    },
    {
        "id": 4,
        "tag": "劳动法-加班费",
        "question": "公司强制要求我们签自愿放弃加班费的协议，这合法吗？",
        "expected_tool": "search_labor_law",
        "standard_cot": "加班费权利是否可被协议放弃，属于劳动法强制性规定范畴。",
        "standard_call": '{"tool": {"name": "search_labor_law", "arguments": {"query": "加班费 放弃协议 强制性规定"}}}'
    },
    {
        "id": 5,
        "tag": "负样本-无需调用工具",
        "question": "你好，请问你能帮我做什么？",
        "expected_tool": None,   # 期望 tool 为 null，不调用工具
        "standard_cot": "用户是日常问候，无法律诉求，无需调用 MCP 工具，直接回复即可。",
        "standard_call": '{"tool": null}'
    },
]


# ── 解析工具调用输出 ──────────────────────────────────────────────────
def parse_output(raw: str) -> tuple[str | None, str]:
    """
    解析模型输出，返回 (tool_name, parse_method)。
    tool_name: 解析到的工具名，无需调用工具时为 None，解析完全失败为 "__parse_failed__"
    parse_method: "json_exact" | "regex_fallback" | "failed"
    """
    cleaned = raw.strip()

    # 剥离 Markdown 标记
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].strip()

    # 1. 标准 JSON 解析
    try:
        data = json.loads(cleaned)
        tool = data.get("tool")
        if tool is None:
            return None, "json_exact"
        return tool["name"], "json_exact"
    except Exception:
        pass

    # 2. 降级：正则提取工具名
    match = re.search(r'(search_(?:labor|civil)_law)', cleaned)
    if match:
        return match.group(1), "regex_fallback"

    # 3. 检查是否输出了 null（负样本场景）
    if re.search(r'"tool"\s*:\s*null', cleaned) or "无需调用" in cleaned:
        return None, "json_exact"

    return "__parse_failed__", "failed"


# ── 单模型测试 ─────────────────────────────────────────────────────────
def test_model(model, tokenizer, model_name: str):
    print(f"\n{'='*60}")
    print(f"  测试模型：{model_name}")
    print('='*60)

    correct_json   = 0  # JSON 精确匹配命中
    correct_regex  = 0  # 降级正则命中
    total = len(test_set)

    for item in test_set:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{item['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids      = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        input_len      = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False
            )

        result = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        parsed_tool, method = parse_output(result)

        # 判断是否命中
        expected = item["expected_tool"]
        if method == "failed":
            hit = False
        elif expected is None:
            hit = (parsed_tool is None)
        else:
            hit = (parsed_tool == expected)

        if hit and method == "json_exact":
            correct_json += 1
        elif hit and method == "regex_fallback":
            correct_regex += 1

        # 逐题打印
        hit_label = "✓" if hit else "✗"
        method_label = {"json_exact": "[JSON]", "regex_fallback": "[正则]", "failed": "[失败]"}[method]
        print(f"\n  【题{item['id']}】{item['tag']}")
        print(f"  问题：{item['question']}")
        print(f"  期望工具：{expected}")
        print(f"  解析工具：{parsed_tool}  {method_label}")
        print(f"  模型原始输出：{result[:120]}")
        print(f"  结果：{hit_label} {'正确' if hit else '错误'}")

    total_hit = correct_json + correct_regex
    print(f"\n  总准确率：  {total_hit}/{total} = {total_hit/total*100:.1f}%")
    print(f"  JSON精确：  {correct_json}/{total}  （输出格式完全规范）")
    print(f"  正则降级：  {correct_regex}/{total}  （工具名正确但格式不规范）")
    print('='*60)


# ── 入口 ───────────────────────────────────────────────────────────────
def main():
    base_model_path  = r"D:\py\Qwen2.5-1.5B"
    lora_path        = r"D:\py\Agent_Distill\qwen_mcp_lora_output"
    merged_path      = r"D:\py\Agent_Distill\qwen_merged"

    # 测试原始模型
    print("加载原始模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    raw_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    raw_model.eval()
    test_model(raw_model, tokenizer, "原始 Qwen2.5-1.5B")
    del raw_model
    torch.cuda.empty_cache()

    # 测试微调后模型（优先用 merged，没有则加载 LoRA）
    print("\n加载微调后模型...")
    import os
    if os.path.exists(merged_path) and len(os.listdir(merged_path)) > 0:
        tuned_model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model_label = "微调后 Qwen2.5-1.5B（merged）"
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tuned_model = PeftModel.from_pretrained(base, lora_path)
        model_label = "微调后 Qwen2.5-1.5B（LoRA）"

    tuned_model.eval()
    test_model(tuned_model, tokenizer, model_label)
    del tuned_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()