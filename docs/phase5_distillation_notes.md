# Phase 5：Qwen3.5-4B Serving + Agent Trajectory 蒸馏

## 目标

在 Phase 4 LangGraph Runtime 上接入 **Qwen3.5-4B**，把教师模型产生的 Agent 执行轨迹转化为结构化监督数据，并通过 LoRA 将路由、工具调用、证据使用和最终回答能力迁移到本地模型。

## 实现内容

- `model_service/qwen35.py`：Qwen3.5-4B 本地 Serving，支持 Raw 与 LoRA adapter。
- `model_service/server.py`：FastAPI / OpenAI-compatible 子集。
- `agent/runtime/qwen35_generator.py`：Runtime Generation Adapter。
- `distill/trajectory.py`：Structured Agent Trajectory。
- `distill/hard_mining.py`：Hard Example Mining。
- `distill/train_phase5.py`：Qwen3.5-4B 4-bit + LoRA SFT。

训练仍采用 NF4、LoRA、gradient checkpointing、batch size 1 + gradient accumulation 8 的单卡方案；目标是蒸馏可观测 Agent 行为，不复制隐藏 CoT。

## Phase 5 / 6 对照实验

Phase 6 的唯一主实验为：

- **Qwen3.5-4B Raw**：原始 Qwen3.5-4B，不加载项目 LoRA。
- **Qwen3.5-4B LoRA**：同一个 Qwen3.5-4B base checkpoint + Phase 5 LoRA adapter。

两者使用相同 Benchmark、Runtime、RAG corpus、Tool Contract、generation 配置和评测代码。
