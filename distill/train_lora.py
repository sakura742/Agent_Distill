"""Phase 5 QLoRA/LoRA training entry point.

Consumes chat-format JSONL produced by ``distill.format_sft`` and saves a PEFT
adapter. The script is intentionally separate from the Phase 4 LangGraph runtime.

The default configuration is conservative for an 8 GB consumer GPU. QLoRA is
optional: ``--load-in-4bit`` requires a working bitsandbytes installation; use
``--no-load-in-4bit`` for a LoRA-only smoke test if 4-bit loading is unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_records(path: Path):
    from datasets import Dataset

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("messages"), list):
                raise ValueError(f"Line {line_no}: missing messages list")
            rows.append(item)
    if not rows:
        raise ValueError(f"Training dataset is empty: {path}")
    return Dataset.from_list(rows)


def _load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--load-in-4bit requires a compatible bitsandbytes installation. "
                "For the first Windows smoke test, use --no-load-in-4bit."
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    return model, tokenizer


def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig
    from transformers import TrainingArguments
    from trl import SFTTrainer

    dataset = _load_records(Path(args.train_file))
    model, tokenizer = _load_model_and_tokenizer(args)

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        max_length=args.max_seq_length,
    )

    result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = dict(result.metrics)
    metrics.update(
        {
            "base_model": str(args.model),
            "train_file": str(args.train_file),
            "load_in_4bit": args.load_in_4bit,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules.split(","),
        }
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLoRA/LoRA fine-tune a local legal Agent model")
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
