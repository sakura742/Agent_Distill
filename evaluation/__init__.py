"""evaluation/ —— 模型评估（原 distill/evaluate.py）。

对比原始 Qwen2.5-1.5B 与微调后模型的工具调用准确率（JSON 精确匹配 + 正则降级
两层评估）。Phase 1 只做搬迁 + import/路径修复，测试集内容、评估逻辑、判定标准
均未改变。
"""
