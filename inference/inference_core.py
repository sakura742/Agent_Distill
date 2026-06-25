import os
import sys
import json
import re
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"

# ── 路径配置 ────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_DIR   = os.path.abspath(os.path.join(_THIS_DIR, "..", "legal_rag", "chroma_db"))

BASE_MODEL_PATH = r"D:\py\Qwen2.5-1.5B"
LORA_PATH       = r"D:\py\Agent_Distill\qwen_mcp_lora_output"
RAW_MODEL_PATH  = r"D:\py\Qwen2.5-1.5B"

MAX_HISTORY_TURNS = 3
LAW_SNIPPET_LIMIT = 600

_LABOR_KEYWORDS = {"工资", "老板", "裁员", "加班", "劳动", "辞退", "社保", "合同", "解雇"}


# ══════════════════════════════════════════════════════════════
# 1. 启动时加载（仅 tokenizer + retriever，不占 GPU）
# ══════════════════════════════════════════════════════════════

def load_models() -> dict:
    """
    启动时调用一次。只加载 tokenizer 和 Chroma retriever，不加载 LLM。
    LLM 在每次推理时按需加载、用完立即释放（8G 单卡方案）。
    """
    print("【inference_core】加载 Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print("【inference_core】加载 Chroma 向量库（CPU）...", flush=True)
    if not (os.path.exists(_DB_DIR) and len(os.listdir(_DB_DIR)) > 0):
        raise FileNotFoundError(f"向量库不存在：{_DB_DIR}，请先运行 legal_rag/ingest.py")

    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(persist_directory=_DB_DIR, embedding_function=embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("【inference_core】Tokenizer + 向量库就绪 ✅", flush=True)
    return {
        "tokenizer": tokenizer,
        "retriever": retriever,
    }


# ══════════════════════════════════════════════════════════════
# 2. 按需加载 / 释放 LLM
# ══════════════════════════════════════════════════════════════

def _load_tuned_model():
    print("【inference_core】加载微调模型...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_PATH, device_map={"": "cuda:0"})
    model.eval()
    return model


def _load_raw_model():
    print("【inference_core】加载原始模型...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        RAW_MODEL_PATH,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model.eval()
    return model


def _release_model(model):
    del model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# 3. 工具函数
# ══════════════════════════════════════════════════════════════

def _generate(model, tokenizer, prompt: str, max_new_tokens: int,
              do_sample: bool = False, **kwargs) -> str:
    device = next(model.parameters()).device
    encoded        = tokenizer(prompt, return_tensors="pt")
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    input_len      = input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            **kwargs,
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def _normalize_tool_name(name: str) -> str:
    if "labor" in name:
        return "search_labor_law"
    if "civil" in name:
        return "search_civil_law"
    return name


def _parse_tool_call(raw: str):
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].strip()

    try:
        data = json.loads(cleaned)
        tool = data.get("tool")
        if tool is None:
            return None
        return _normalize_tool_name(tool["name"]), tool["arguments"]["query"]
    except Exception:
        pass

    tool_m  = re.search(r'(search_(?:labor|civil)_law)', cleaned)
    query_m = re.search(r'["\']query["\']\s*:\s*["\'](.*?)["\']', cleaned)
    if not query_m:
        query_m = re.search(r'keyword=["\'](.*?)["\']', cleaned)
    if tool_m and query_m:
        return _normalize_tool_name(tool_m.group(1)), query_m.group(1)

    return None


def _build_history_prompt(history: list) -> str:
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    parts = []
    for turn in trimmed:
        parts.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
    return "\n".join(parts)


def _extract_thought(raw: str) -> str:
    try:
        return json.loads(raw).get("thought", "")
    except Exception:
        return raw[:100]


def _rag_search(retriever, query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


# ══════════════════════════════════════════════════════════════
# 4. 推理流程
# ══════════════════════════════════════════════════════════════

_STAGE1_SYSTEM = (
    "你是一个熟练对接后端法律知识库的智能助手。"
    "请分析用户的法律诉求，给出思考过程，并调用正确的工具检索法条。\n"
    "必须严格按照以下 JSON 格式输出，不要输出其他任何内容：\n"
    '{"thought": "你的分析过程", "tool": {"name": "工具名", "arguments": {"query": "检索关键词"}}}\n'
    "可用工具：search_labor_law（劳动法相关）、search_civil_law（民法相关）。\n"
    "如无需调用工具，tool 字段填 null。"
)

_STAGE3_SYSTEM = (
    "你是一个专业的常驻法律顾问。"
    "请根据用户的问题和对话历史，严格结合以下检索到的法律条款，"
    "给用户提供有法可依、条理清晰的维权行动指南。"
    "回答中必须明确引用具体条款编号和赔偿标准。"
)

_RAW_SYSTEM = (
    "你是一位经验丰富的中国法律顾问，熟悉《民法典》《劳动合同法》《劳动法》等法律法规。"
    "请根据用户描述的情况，结合你掌握的法律知识，给出条理清晰、有据可依的专业建议。"
    "如果涉及具体法条，请尽量说明对应的法律名称和大致内容。"
)


def _run_tuned(tokenizer, retriever, user_query: str, history: list) -> dict:
    """加载微调模型 → 三阶段推理 → 立即释放显存。"""
    model = _load_tuned_model()

    # 阶段一：工具调用决策
    stage1_prompt = (
        f"<|im_start|>system\n{_STAGE1_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    agent_output = _generate(model, tokenizer, stage1_prompt, max_new_tokens=256)

    # 阶段二：检索法条
    parsed = _parse_tool_call(agent_output)
    if parsed:
        tool_name, query = parsed
    else:
        tool_name = "search_labor_law" if any(k in user_query for k in _LABOR_KEYWORDS) else "search_civil_law"
        query     = user_query
    db_result = _rag_search(retriever, query)[:LAW_SNIPPET_LIMIT]

    # 阶段三：RAG 增强生成（含多轮历史）
    history_block = _build_history_prompt(history)
    if history_block:
        history_block = "\n" + history_block
    stage3_prompt = (
        f"<|im_start|>system\n{_STAGE3_SYSTEM}<|im_end|>"
        f"{history_block}\n"
        f"<|im_start|>user\n用户问题：{user_query}\n\n参考法条：\n{db_result}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    answer = _generate(model, tokenizer, stage3_prompt,
                       max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)

    _release_model(model)
    return {
        "answer": answer,
        "reasoning": {
            "thought":     _extract_thought(agent_output),
            "tool":        tool_name,
            "query":       query,
            "law_snippet": db_result[:LAW_SNIPPET_LIMIT],
        },
    }


def _run_raw(tokenizer, user_query: str, history: list) -> str:
    """加载原始模型 → 推理 → 立即释放显存。"""
    model = _load_raw_model()

    history_block = _build_history_prompt(history)
    if history_block:
        history_block = "\n" + history_block
    prompt = (
        f"<|im_start|>system\n{_RAW_SYSTEM}<|im_end|>"
        f"{history_block}\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    answer = _generate(model, tokenizer, prompt,
                       max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)

    _release_model(model)
    return answer


# ══════════════════════════════════════════════════════════════
# 5. 统一对外接口
# ══════════════════════════════════════════════════════════════

def run_inference(resources: dict, user_query: str, history: list = None) -> dict:
    """
    串行推理两个模型，返回双列结果。

    参数：
        resources:   load_models() 的返回值
        user_query:  当前用户输入
        history:     [{"role": "user"/"assistant", "content": "..."}]，None 表示单轮

    返回：
    {
        "raw_answer":   str,
        "tuned_answer": str,
        "reasoning": {
            "thought":     str,
            "tool":        str,
            "query":       str,
            "law_snippet": str
        }
    }
    """
    if history is None:
        history = []

    tokenizer = resources["tokenizer"]
    retriever = resources["retriever"]

    # 先跑微调模型，释放后再跑原始模型
    tuned_result = _run_tuned(tokenizer, retriever, user_query, history)
    raw_answer   = _run_raw(tokenizer, user_query, history)

    return {
        "raw_answer":   raw_answer,
        "tuned_answer": tuned_result["answer"],
        "reasoning":    tuned_result["reasoning"],
    }


# ══════════════════════════════════════════════════════════════
# 6. 本地调试
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    resources = load_models()

    # 单轮
    result = run_inference(
        resources,
        user_query="老板突然把我踢出工作群，还扣了我半个月工资，我该怎么维权？",
    )
    print("\n=== 原始模型 ===")
    print(result["raw_answer"])
    print("\n=== 微调模型 ===")
    print(result["tuned_answer"])
    print("\n=== 推理过程 ===")
    print(json.dumps(result["reasoning"], ensure_ascii=False, indent=2))

    # 多轮追问
    history = [
        {"role": "user",      "content": "老板突然把我踢出工作群，还扣了我半个月工资，我该怎么维权？"},
        {"role": "assistant", "content": result["tuned_answer"]},
    ]
    result2 = run_inference(resources, user_query="赔偿金具体怎么计算？", history=history)
    print("\n=== 多轮追问：微调模型 ===")
    print(result2["tuned_answer"])