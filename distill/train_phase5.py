"""Phase 5: aligned Qwen3.5-4B LoRA SFT.

Usage:
  uv run python -m distill.train_phase5 --mode decision
  uv run python -m distill.train_phase5 --mode answer
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from configs.settings import settings


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"训练数据不存在: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"训练数据为空: {path}")
    return rows


def _to_text(row: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )


def _trajectory_to_text(row: dict, tokenizer) -> str:
    """Backward-compatible serializer used by legacy Phase 5 tests/tools.

    New training data is intentionally split into decision and answer targets;
    this helper remains available so old callers do not break during migration.
    """
    messages = [
        {"role": "system", "content": "你是法律 Agent。输出可观察的 Agent 行为。"},
        {"role": "user", "content": row.get("question", "")},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "domain": row.get("domain"),
                    "intent": row.get("intent"),
                    "plan": row.get("plan", []),
                    "tool": row.get("tool", {}),
                    "citations": row.get("citations", []),
                    "answer": row.get("answer", ""),
                },
                ensure_ascii=False,
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train(mode: str) -> None:
    if mode not in {"decision", "answer"}:
        raise ValueError("mode 必须是 decision 或 answer")
    data_path = (
        settings.phase5_decision_data_path
        if mode == "decision"
        else settings.phase5_answer_data_path
    )
    output_dir = (
        settings.qwen35_decision_lora_output_dir
        if mode == "decision"
        else settings.qwen35_answer_lora_output_dir
    )
    model_path = settings.qwen35_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=settings.hf_local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = _load_rows(data_path)
    dataset = Dataset.from_list([{"text": _to_text(r, tokenizer)} for r in rows])
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=settings.hf_local_files_only,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    peft = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=3,
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
        peft_config=peft,
        processing_class=tokenizer,
        args=args,
    )
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Phase 5 {mode} LoRA 已保存：{output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decision", "answer"], required=True)
    train(parser.parse_args().mode)
