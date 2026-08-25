"""tests/ —— Phase 1 新增的基础测试目录。

当前只包含不依赖 GPU / 模型权重 / 重量级三方库（torch、transformers、chromadb ...）
的"结构性"测试：配置系统能否正确加载、各模块能否被正常 import、关键路径变量类型
是否正确。训练 / 推理 / RAG 检索这些需要真实模型权重和显卡的流程，测试覆盖见
docs/legacy 之外新增的 `tests/README.md` 中的说明。
"""
