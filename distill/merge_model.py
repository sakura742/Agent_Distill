"""LoRA 权重合并工具（原根目录 merge_model.py，Phase 1 迁移至 distill/，
因为它是蒸馏产物的后处理步骤，和 train.py / gen_data.py 同属蒸馏管线）。

合并逻辑（CPU 上 PeftModel.merge_and_unload()）完全未变，只是三个路径改为
从 configs.settings 读取。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from configs.settings import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

base_model_path = settings.base_model_path
lora_path = str(settings.lora_output_dir)
output_path = str(settings.merged_model_dir)

logger.info("加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"  # 合并在CPU做，省显存
)

logger.info("加载LoRA权重...")
model = PeftModel.from_pretrained(model, lora_path)

logger.info("合并中...")
model = model.merge_and_unload()

logger.info("保存合并后模型...")
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

logger.info("完成！合并模型保存至 %s", output_path)
