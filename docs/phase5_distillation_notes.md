# Phase 5：Qwen3.5-2B Serving + Agent Trajectory 蒸馏

## 目标

在 Phase 4 LangGraph Runtime 上接入 Qwen3.5-2B，并把教师模型产生的 Agent 执行轨迹转化为结构化监督数据，再通过 LoRA 将路由、工具调用、证据使用和最终回答能力迁移到本地小模型。

Qwen3.5-2B 提供 Transformers 格式权重，并可通过 Transformers、vLLM、SGLang 等方式部署；本项目 Phase 5 先采用 Transformers + FastAPI 的本地 Serving，避免把 Runtime 与具体推理框架耦合。

## 架构

```text
                         Teacher / DeepSeek
                                │
                                ▼
User Question → Phase 4 Runtime → Tool Service → RAG
                     │                         │
                     └──────────────┬──────────┘
                                    ▼
                           Structured Trajectory
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                Hard Example Mining        Normal Samples
                       └────────────┬────────────┘
                                    ▼
                         Qwen3.5-2B + LoRA SFT
                                    │
                                    ▼
                         Local Qwen3.5 Serving
                                    │
                                    ▼
                         Phase 4 Agent Runtime
```

## 实现内容

### 1. Qwen3.5 Serving

- `model_service/qwen35.py`
  - 单进程常驻加载模型。
  - `apply_chat_template` 统一消息格式。
  - GPU 自动选择；CUDA 使用 BF16 推理。
  - 线程锁保证单卡模型生成阶段不会发生并发写状态问题。
- `model_service/server.py`
  - FastAPI 服务。
  - `/health` 健康检查。
  - `/v1/chat/completions` OpenAI-compatible 子集。
- `agent/runtime/qwen35_generator.py`
  - 将 Qwen3.5 Serving 适配成 Phase 4 `answer_generator`。
  - Runtime 不直接依赖模型加载细节。

### 2. Trajectory 蒸馏

`distill/trajectory.py` 通过 Phase 4 Graph 生成结构化轨迹。保存字段：

- `question`
- `domain / intent / intent_confidence`
- `plan`
- `tool.name / tool.arguments`
- `retrieved_documents / citations`
- `answer`
- `verification`
- `retry_count`
- `trace`

这里不把隐藏 CoT 作为训练目标，而是蒸馏可观测的 Agent 行为轨迹，避免把教师模型的冗长思维过程直接复制给学生。

### 3. Hard Example Mining

`distill/hard_mining.py` 根据 Runtime 信号自动筛选困难样本：

- Verification 失败。
- 发生 Re-plan / retry。
- Router confidence < 0.55。
- 工具名或查询参数缺失。
- 缺少引用证据。

困难样本单独保存到 `hard_examples.jsonl`，训练时与普通 trajectory 合并去重。

### 4. Qwen3.5 LoRA

`distill/train_phase5.py` 使用：

- Qwen3.5-2B。
- 4-bit NF4 Quantization。
- LoRA `r=8 / alpha=16 / dropout=0.05`。
- Gradient Checkpointing。
- batch size 1 + gradient accumulation 8。
- max length 2048。
- 训练目标为结构化 `domain → intent → plan → tool → evidence → answer`。

该配置面向单张 8 GB GPU 的资源约束；如果显存不足，优先降低 `max_length`，再降低 LoRA rank。

## 数据流

```text
原始问题
   ↓
Hybrid Router
   ↓
Task Plan
   ↓
Tool Decision
   ↓
Tool Execution
   ↓
RAG / Citations
   ↓
Teacher Answer
   ↓
Verification
   ↓
trajectory.jsonl
   ↓
hard_mining.py
   ↓
LoRA SFT
```

## 运行顺序

### A. 配置

设置：

- `AGENT_DISTILL_QWEN35_MODEL_PATH`
- `AGENT_DISTILL_QWEN35_LORA_OUTPUT_DIR`
- `DEEPSEEK_API_KEY`

### B. 生成轨迹

```bash
python -m distill.trajectory data/questions.jsonl
```

### C. 挖掘困难样本

```bash
python -m distill.hard_mining
```

### D. LoRA

```bash
python -m distill.train_phase5
```

### E. 启动 Serving

```bash
uvicorn model_service.server:app --host 0.0.0.0 --port 8000
```

### F. 接入 Runtime

```python
from agent.runtime.graph import build_legal_agent_graph
from agent.runtime.qwen35_generator import Qwen35AnswerGenerator

graph = build_legal_agent_graph(answer_generator=Qwen35AnswerGenerator())
```

## Phase 5 完成标准

- [x] Qwen3.5-2B 配置独立于 Qwen2.5-1.5B。
- [x] Qwen3.5 本地 Serving。
- [x] FastAPI/OpenAI-compatible 推理入口。
- [x] Runtime Generation Adapter。
- [x] Structured Agent Trajectory 数据格式。
- [x] Hard Example Mining。
- [x] 4-bit + LoRA 训练脚本。
- [x] Phase 5 配置和依赖隔离。
- [ ] Benchmark 数值对比 —— 留给 Phase 6。
- [ ] Checkpoint / 多轮会话 —— 留给 Phase 7。

## 边界

Phase 5 不负责建立完整 Benchmark，也不在本阶段修改 Phase 4 的 Router / Retrieval 算法。Phase 6 将基于这些 trajectory 和 Runtime trace 建立 Routing、Retrieval、Tool Calling、Workflow、Answer、Multi-turn 六类 Benchmark。
