"""Configuration for Phase 5 LoRA fine-tuning.

All model/data paths are supplied explicitly so training remains reproducible and does
not depend on machine-specific absolute paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoRAConfig:
    base_model_path: Path
    train_file: Path
    output_dir: Path
    max_seq_length: int = 2048
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

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
