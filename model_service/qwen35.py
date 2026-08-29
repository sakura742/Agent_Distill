"""Qwen3.5-4B local serving adapter."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.settings import settings


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False


class Qwen35Service:
    """Thread-safe local Qwen3.5-4B service with optional LoRA adapter."""

    def __init__(self, model_path: str | None = None, adapter_path: str | None = None) -> None:
        self.model_path = model_path or settings.qwen35_model_path
        self.adapter_path = adapter_path
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=settings.hf_local_files_only
        )
        kwargs: dict[str, Any] = {
            "device_map": "auto", "low_cpu_mem_usage": True,
            "trust_remote_code": True, "local_files_only": settings.hf_local_files_only,
            "attn_implementation": "sdpa",
        }
        if torch.cuda.is_available():
            # 与 distill/train_phase5.py 的量化配置保持一致：LoRA adapter 就是在
            # 4bit NF4 基座上训练出来的，评估时不量化会在 8GB 显存下触发
            # accelerate 的 CPU offload（大量层间数据搬运），导致推理极慢。
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path, is_trainable=False)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        with self._lock:
            self.load()
            assert self._tokenizer is not None and self._model is not None
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self._tokenizer(prompt, return_tensors="pt")
            device = next(self._model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                output = self._model.generate(
                    **encoded, max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature, top_p=config.top_p,
                    do_sample=config.do_sample,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            generated = output[0][encoded["input_ids"].shape[1]:]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def health(self) -> dict[str, Any]:
        return {
            "model": self.model_path,
            "adapter": self.adapter_path,
            "loaded": self.loaded,
            "device": str(next(self._model.parameters()).device) if self._model else None,
        }
