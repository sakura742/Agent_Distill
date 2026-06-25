# Agent_Distill

用教师模型（DeepSeek）蒸馏法律场景 Agent 能力到 Qwen2.5-1.5B，结合本地 RAG 法律知识库实现完整的工具调用闭环。

---

## 目录结构

```
Agent_Distill/
├── legal_rag/
│   ├── ingest.py          # 解析法律 PDF，分块写入 Chroma 向量库（一次性运行）
│   ├── server.py          # MCP 服务端，暴露 search_law 检索工具
│   └── debug_rag.py       # 验证 MCP 链路连通性
│
├── distill/
│   ├── tools_config.json  # 工具定义（search_civil_law / search_labor_law）
│   ├── gen_data.py        # 调用 DeepSeek API 生成 CoT + 工具调用训练数据
│   ├── train.py           # LoRA 微调 Qwen2.5-1.5B（INT8 量化，约 6G 显存）
│   └── evaluate.py        # 微调前后工具调用准确率对比
│
├── inference/
│   ├── agent_pipeline.py  # 三阶段闭环：Qwen 决策 → 工具调用 → RAG 增强回答
│   └── compare_demo.py    # 五个场景并排对比原始模型 vs 微调模型 vs RAG增强
│
├── merge_model.py         # 将 LoRA 权重合并进基座，导出完整模型
└── data/
    ├── labor_law.pdf
    └── minfa.pdf
```

## 运行顺序

```bash
python legal_rag/ingest.py       # 1. 建库（一次性）
python distill/gen_data.py       # 2. 生成训练数据（需 DeepSeek API Key）
python distill/train.py          # 3. 微调（需 GPU）
python distill/evaluate.py       # 4. 评估效果
python inference/compare_demo.py # 5. 查看对比演示
```

## 技术栈

| 用途 | 方案 |
|---|---|
| 基座模型 | Qwen2.5-1.5B |
| 教师模型 | DeepSeek V3 API |
| 微调 | LoRA（PEFT + TRL SFTTrainer） |
| 向量库 | Chroma + text2vec-base-chinese |
| RAG 框架 | LangChain |
| Agent 协议 | MCP（FastMCP） |
