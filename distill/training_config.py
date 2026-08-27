"""Configuration for Phase 5 parameter-efficient fine-tuning.

The configuration is deliberately independent from the LangGraph runtime. Paths are
provided by the CLI so the same training code works on different machines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoRAConfig:
    base_model_path: Path
    train_file: Path
    output_dir: Path
    max_seq_length: int = 1024
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    logging_steps: int = 1
    save_steps: int = 50
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
        if not self.train_file.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_r and lora_alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
