"""distill/ —— Agent Distillation 蒸馏管线（数据生成 + LoRA 训练 + 模型合并）。

包含：
- ``tools_config.json``：工具 JSON Schema（数据生成 prompt 的工具描述来源）。
- ``gen_data``：调用 DeepSeek 生成 CoT + 工具调用训练数据。
- ``train``：LoRA 微调 Qwen2.5-1.5B（INT8 量化 + TRL SFTTrainer）。
- ``merge_model``：将 LoRA 权重合并进基座模型（原根目录 merge_model.py 迁移至此，
  因为它是蒸馏产物后处理的一部分）。
- ``data/agent_distill_train.jsonl``：蒸馏训练数据（原根目录文件迁移至此，
  Alpaca 三段式格式未改变）。

Phase 1 明确要求：训练核心逻辑（LoRA 超参、SFTConfig、formatting 模板）与数据集
格式（Alpaca instruction/input/output 三段式）**完全未修改**，只是把硬编码路径
和硬编码 API Key 换成 configs.settings 读取。
"""
