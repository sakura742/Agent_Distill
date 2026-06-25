import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


base_model_path = r"D:\py\Qwen2.5-1.5B"
lora_path = r"D:\py\Agent_Distill\qwen_mcp_lora_output"
output_path = r"D:\py\Agent_Distill\qwen_merged"

print("加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"  # 合并在CPU做，省显存
)

print("加载LoRA权重...")
model = PeftModel.from_pretrained(model, lora_path)

print("合并中...")
model = model.merge_and_unload()

print("保存合并后模型...")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"完成！合并模型保存至 {output_path}")