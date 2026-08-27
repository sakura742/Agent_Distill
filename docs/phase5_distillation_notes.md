# Phase 5：Agent Distillation

## 目标

把原来的“问题 → 工具调用字符串”训练方式升级为结构化 Agent Trajectory，并让教师生成、数据校验、SFT 转换、参数高效微调和推理适配彼此解耦。

Phase 5 的验收对象不是单独的 LoRA 训练脚本，而是一条可复现的 Agent Distillation Pipeline：

```text
Phase 4 LangGraph Runtime
        ↓
Runtime-backed Teacher Trajectory
        ↓
Trajectory Validation
        ↓
SFT Formatting
        ↓
Qwen3.5 base model
        ↓
QLoRA / LoRA
        ↓
Adapter Checkpoint
        ↓
Agent Inference Adapter
        ↓
Phase 4 Runtime
        ↓
Phase 6 Benchmark
```

## 已完成

- `distill/trajectory_schema.py`：统一 `AgentTrajectory / AgentStep / ToolCall` 契约。
- `distill/teacher_pipeline.py`：DeepSeek 教师模型结构化生成轨迹。
- Runtime-backed collector：教师决定工具后执行真实工具并记录 observation，再继续决策。
- `distill/validate_dataset.py`：批量校验 JSONL 轨迹，报告缺失字段和类型错误。
- `distill/format_sft.py`：将 canonical trajectory 转换为 chat-format SFT JSONL。
- `distill/model_adapter.py`：隔离 Transformers/PEFT 加载细节，为 LangGraph 提供轻量本地模型接口。
- `distill/training_config.py`：统一 QLoRA/LoRA 训练参数。
- `distill/train_lora.py`：基于 Transformers 5.x + PEFT 0.19 + TRL 1.5 的训练入口，支持 QLoRA 与 LoRA 两种模式。

## 数据设计

每条 canonical trajectory 包含：

- `user_query`
- `intent`
- `domain`
- `steps[]`
  - `action`
  - `tool_call`
  - `observation`
- `final_answer`
- `metadata`

训练数据不把工具调用埋在一段不可解析的自然语言中，而是先保留结构化轨迹，再转换为 SFT 文本。训练监督重点放在可观测 Agent 行为：意图/法域判断、工具选择、参数生成、Observation 处理、任务完成和最终回答；不要求模型复现隐藏 chain-of-thought。

## Runtime-backed Teacher

Phase 4 的 LangGraph Runtime 是执行基准。教师轨迹不能长期依赖模型凭空描述 observation，而应遵循：

```text
Teacher decision
      ↓
Tool validation
      ↓
Real tool / MCP execution
      ↓
Observation
      ↓
Teacher next decision
      ↓
...
      ↓
Final answer
```

这样生成的数据才能用于学习真实的 Tool Calling 与多步骤任务执行，而不是只学习字符串格式。

## Qwen3.5 模型与 LoRA

目标本地模型已经完成加载 smoke test，并确认模型类型为 `Qwen3_5ForCausalLM`。当前实际 checkpoint 约 4.22B 参数；一次 PEFT 注入测试得到：

```text
all params:       4,219,415,552
trainable params:    13,664,256
trainable ratio:          0.3238%
```

Qwen3.5 的结构不是传统 Qwen2.5 的纯 `q_proj/k_proj/v_proj/o_proj` 形式。实际 checkpoint 同时包含 `linear_attn` 和部分 `self_attn` 层。当前默认 LoRA target modules 为：

```text
in_proj_qkv
in_proj_z
out_proj
gate_proj
up_proj
down_proj
```

该 target 配置已经通过 PEFT 注入 smoke test；正式训练前仍需用目标数据执行 tiny overfit，确认 loss、显存和 adapter 保存/加载均正常。

## 训练模式

默认训练入口采用 QLoRA：4-bit NF4 base model + LoRA adapter + gradient checkpointing。由于 bitsandbytes 在不同 Windows/CUDA 环境的可用性需要单独验证，训练脚本同时支持：

```bash
--load-in-4bit
--no-load-in-4bit
```

如果目标机器无法使用 bitsandbytes，先使用 `--no-load-in-4bit` 做 LoRA smoke test；这不是正式的 8GB 显存配置承诺。

### RTX 4060 8GB 第一轮建议

```text
batch size:             1
gradient accumulation:  8
max sequence length:    1024
epochs:                 1
LoRA r:                 8
LoRA alpha:             16
LoRA dropout:           0.05
learning rate:          2e-4
gradient checkpointing:  on
```

第一轮只使用约 10～30 条高质量 trajectory，目标是验证训练闭环，不用于得出最终效果结论。

## 训练入口

```bash
uv run python -m distill.train_lora \
  --model D:/py/models \
  --train-file data/trajectories/smoke_sft.jsonl \
  --output-dir outputs/qwen3.5-2b-agent-smoke \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 1 \
  --max-seq-length 1024 \
  --gradient-checkpointing
```

若本机尚未配置可用的 bitsandbytes：

```bash
uv run python -m distill.train_lora \
  --model D:/py/models \
  --train-file data/trajectories/smoke_sft.jsonl \
  --output-dir outputs/qwen3.5-2b-agent-smoke \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 1 \
  --max-seq-length 1024 \
  --no-load-in-4bit \
  --gradient-checkpointing
```

## Reproducibility

模型、训练数据和输出目录均通过 CLI 参数指定，不把开发机绝对路径写入代码。训练输出保存 PEFT adapter 与 tokenizer artifacts；base model 作为独立依赖管理。训练结束后额外保存 `training_metrics.json`，记录模型、数据、LoRA 参数和 Trainer 指标。

## 验收标准

Phase 5 不能以“训练脚本成功启动”作为完成标准。必须依次验证：

1. Qwen3.5 base model loading smoke test；
2. PEFT LoRA target injection smoke test；
3. 10～30 条 trajectory 的 tiny overfit/sanity run；
4. loss 正常计算并能够反向传播；
5. adapter checkpoint 创建；
6. adapter 独立重新加载并完成推理；
7. 微调模型重新接入 Phase 4 LangGraph Runtime；
8. 使用与 Base Model 相同的 Benchmark 做 Base vs LoRA 对比；
9. Phase 6 根据 Tool Selection Accuracy、Argument Accuracy、Task Success Rate、Citation Accuracy 等指标分析收益，并将失败样本回流 Hard-example mining。

## 与后续 Phase 的关系

Phase 5 负责“教师轨迹 → 参数高效微调 → Agent adapter”。Phase 6 负责统一 Benchmark、自动化评估和错误分析。两阶段必须使用相同的 Agent Runtime、Tool Schema 和评估数据，才能可信地回答“蒸馏是否真正提升了 Agent 能力”。
