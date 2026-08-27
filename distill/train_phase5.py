"""Phase 5 Qwen3.5-2B LoRA training on structured Agent trajectories.

Designed for a single 8 GB GPU: 4-bit NF4 base model + LoRA + gradient
checkpointing. The target is behavior distillation (routing/tool/evidence/
answer structure), not copying hidden chain-of-thought.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from configs.settings import settings


def _trajectory_to_text(item: dict) -> str:
    tool = item.get("tool", {})
    target = {
        "domain": item.get("domain"),
        "intent": item.get("intent"),
        "plan": item.get("plan", []),
        "tool": {
            "name": tool.get("name"),
            "arguments": tool.get("arguments", {}),
        },
        "evidence": item.get("citations", []),
        "answer": item.get("answer", ""),
    }
    messages = [
        {"role": "system", "content": "你是法律 Agent。先确定法域和工具，再基于检索证据给出可靠答案。"},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _load_data() -> list[dict]:
    paths = [settings.trajectory_data_path]
    if settings.hard_example_path.exists():
        paths.append(settings.hard_example_path)
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        if not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("question", "")
                if key and key not in seen:
                    seen.add(key)
                    rows.append(row)
    if not rows:
        raise FileNotFoundError("没有 trajectory 数据，请先运行 distill/trajectory.py")
    return rows


model_path = settings.qwen35_model_path
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=settings.hf_local_files_only,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def train() -> None:
    rows = _load_data()
    texts = [{"text": _trajectory_to_text(row)} for row in rows]
    dataset = __import__("datasets").Dataset.from_list(texts)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=settings.hf_local_files_only,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = SFTConfig(
        output_dir=str(settings.qwen35_lora_output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=2,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        max_length=2048,
        dataset_num_proc=1,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=args,
    )
    trainer.train()
    Path(settings.qwen35_lora_output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(settings.qwen35_lora_output_dir)
    tokenizer.save_pretrained(settings.qwen35_lora_output_dir)
    print(f"Phase 5 LoRA 已保存：{settings.qwen35_lora_output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train()
