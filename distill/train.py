# -*- coding: utf-8 -*-
#
# Phase 1 重构说明：本文件的训练核心逻辑（量化配置、LoRA 超参、
# formatting_prompts_func 模板、SFTConfig/SFTTrainer 参数）按要求【完全未修改】。
# 唯一改动：
#   1. 三行硬编码的 Windows 绝对路径改为从 configs.settings 读取（不设置任何
#      环境变量时，取值与重构前的硬编码值完全相同，行为不变）；
#   2. 【健壮性修复，非逻辑改动，2026-08-25 二次修正】stdout 重新包装为 UTF-8
#      这件事，从"模块顶层、import 时无条件执行"改为"只在 `python train.py`
#      作为脚本直接运行时（`__name__ == '__main__'`）才执行"。
#      原因：hasattr(sys.stdout, "buffer") 判断本身没错，但 pytest 的 capture
#      对象*恰好也*带有 `.buffer` 属性——当本文件被其他代码 import（例如
#      tests/ 里"能不能被正常 import"的结构性测试）时，旧写法会在 import
#      阶段就把全局 sys.stdout 换成一个包住 pytest 内部 capture buffer 的新
#      TextIOWrapper；等 pytest 会话结束去回收自己的 capture buffer 时，会
#      因为这个 buffer 已经被外部代码"偷梁换柱"过而抛出
#      `ValueError: I/O operation on closed file`，并连带把同一进程里其他
#      测试也弄挂。实测复现：仅仅 `import distill.train` 一次即可让整个
#      pytest 会话在收尾阶段崩溃（1 failed + 4 个无关测试被连带 ERROR）。
#      挪到 `if __name__ == "__main__":` 内部后，import 该模块不再有任何
#      副作用；真正把它当脚本跑（`python distill/train.py`）时行为和以前
#      完全一样，仍然会在真实终端/Windows 控制台下把 stdout 包成 UTF-8。
import sys
import io
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import settings

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


def train():

    model_path = settings.base_model_path
    data_path = str(settings.train_data_path)
    output_dir = str(settings.lora_output_dir)

    print("【1】加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    print("【2】加载 Qwen 基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print("【3】注入 LoRA Adapter...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    print("【4】加载训练数据集...")
    dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train"
    )

    def formatting_prompts_func(example):
        text = (
            f"<|im_start|>system\n"
            f"{example['instruction']}"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{example['input']}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{example['output']}"
            f"<|im_end|>"
        )
        return text

    print("【5】配置 SFTConfig...")
    training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,             # 把 fp16=True 改成 bf16=True
    report_to="none",
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    max_length=1024,
    dataset_num_proc=1,
    packing=False,
)
    print("【6】构建 SFTTrainer...")
    trainer = SFTTrainer(
    model=model,               # 传原始模型，不要先 get_peft_model
    train_dataset=dataset,
    peft_config=peft_config,   # SFTTrainer 自己注入 LoRA
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_args,
)

    print("【7】开始 LoRA 微调训练...")
    trainer.train()

    print("【8】保存 LoRA 权重...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("训练完成！")
    print(f"LoRA 已保存至: {output_dir}")


if __name__ == "__main__":
    # 仅在作为脚本直接运行时才重新包装 stdout 为 UTF-8（原因见文件顶部说明），
    # 避免 import 该模块产生副作用。
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    train()