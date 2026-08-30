"""LoRA 权重合并工具。

重要：本模块只能在显式执行 ``main()`` 时加载模型，不能在 import 时加载。
这样 pytest、web 服务和其它业务模块不会因为本机模型路径不可用而 import 失败。
"""

from __future__ import annotations

import os


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.logging_config import get_logger
    from configs.settings import settings

    logger = get_logger(__name__)
    base_model_path = settings.base_model_path
    lora_path = str(settings.lora_output_dir)
    output_path = str(settings.merged_model_dir)

    logger.info("加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
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


if __name__ == "__main__":
    main()
