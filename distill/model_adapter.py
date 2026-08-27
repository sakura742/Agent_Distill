"""Local causal-LM adapter used by Phase 5 inference and evaluation.

The adapter keeps model loading behind a small interface so LangGraph does not depend on
Transformers/PEFT details. It supports a base model path and an optional LoRA adapter path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    do_sample: bool = False


class LocalAgentModel:
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True, "device_map": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        config = config or GenerationConfig()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
