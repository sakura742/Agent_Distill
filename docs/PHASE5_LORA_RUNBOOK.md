# Phase 5 LoRA 开训交接手册

> 分支：`fix/phase6`。本手册用于从稳定的 Retrieval baseline 进入 Qwen3.5-4B LoRA 实验。

## 1. 开训决策

Retrieval 阶段暂时冻结，不再为了单个 #007 hard case 无限修改。

当前实验基线：

```text
Qwen3.5-4B Raw
GTE-large-zh retrieval
Reranker = OFF
Hybrid = OFF
Rewrite = A/B capability；第一轮 LoRA 推荐使用纯 GTE baseline
```

#007“邻居漏水”仍为 Retrieval hard case。其 gold 未进入 candidate pool，因此不能靠 Reranker 修复；保留为 evaluation/hard example，不把失败检索轨迹作为正常监督样本。

## 2. 训练数据原则

严禁直接使用历史 `distill/data/agent_trajectory.jsonl` 作为最终 SFT 数据，除非它是用冻结后的 Runtime 重新生成并通过过滤。

重新生成后：

```text
agent_trajectory_v2.jsonl
        ↓
distill/filter_trajectory.py
        ├── accepted
        └── rejected/quarantine
        ↓
distill/prepare_phase5_data.py
        ├── phase5_decision.jsonl
        └── phase5_answer.jsonl
        ↓
distill/audit_phase5_data.py
        ↓
LoRA
```

合法的 non-legal/unknown 轨迹必须保留为 Decision SFT 负路由监督：

```json
{"domain":"unknown", "tool":{"name":null}}
```

空 answer、verification failed、unknown 却调用法律工具、legal 却缺少合法工具的轨迹必须进入 rejected。

## 3. 重新生成 trajectory v2

使用冻结的 GTE index。推荐在当前 PowerShell 会话显式设置：

```powershell
$env:AGENT_DISTILL_EMBEDDING_MODEL="thenlper/gte-large-zh"
$env:AGENT_DISTILL_CHROMA_DB_DIR="knowledge/chroma_gte_large_zh"
$env:AGENT_DISTILL_RETRIEVAL_RERANK="0"
$env:AGENT_DISTILL_RETRIEVAL_QUERY_REWRITE="0"
```

然后：

```powershell
uv run python -m distill.trajectory `
  distill/data/phase5_raw_questions.jsonl `
  --output distill/data/agent_trajectory_v2.jsonl
```

默认覆盖旧输出，不使用 `--append`，除非明确要增量生成。

## 4. 过滤失败轨迹

```powershell
uv run python -m distill.filter_trajectory `
  distill/data/agent_trajectory_v2.jsonl `
  --accepted distill/data/agent_trajectory_v2_accepted.jsonl `
  --rejected distill/data/agent_trajectory_v2_rejected.jsonl
```

期望 accepted 同时包含：

- legal + 正确 tool + 非空 answer + verification pass；
- unknown/non-legal + 无 tool + 非空 answer + verification pass。

## 5. 准备 Phase 5 数据

当前 `prepare_phase5_data.py` 使用 `settings.trajectory_data_path`。推荐通过环境变量把它指向 accepted 数据：

```powershell
$env:AGENT_DISTILL_TRAJECTORY_DATA_PATH="D:\py\legal_agent\Agent_Distill\distill\data\agent_trajectory_v2_accepted.jsonl"
uv run python -m distill.prepare_phase5_data
```

生成：

```text
phase5_decision.jsonl
phase5_answer.jsonl
```

## 6. 训练前审计

```powershell
uv run python -m distill.audit_phase5_data `
  distill/data/phase5_decision.jsonl `
  distill/data/phase5_answer.jsonl
```

`audit_phase5_data.py` 同时支持：

- validated trajectory JSONL；
- `prepare_phase5_data.py` 生成的 `{"messages": [...]}` ChatML JSONL。

它会检查：

```text
rows > 0
empty_question = 0
empty_answer = 0
duplicate_questions = 0
```

任何失败都应停止训练。

## 7. 第一轮 LoRA

当前 `distill/train_phase5.py` 已提供：

```powershell
uv run python -m distill.train_phase5 --mode decision
uv run python -m distill.train_phase5 --mode answer
```

使用 Qwen3.5-4B、4-bit quantization、LoRA。第一轮保持既定超参数，不在数据问题未排除前调参。

注意：LoRA 权重目录不要提交到 Git；只保留代码、配置和必要的小型评估结果。

## 8. 训练后的验收

不要只看 train loss。至少重新跑：

```text
Router / domain accuracy
Tool Selection Accuracy
Answer quality
Citation grounding
Verification validity
Task Success Rate
```

核心比较：

```text
Qwen3.5-4B Raw
        vs
Qwen3.5-4B + Decision LoRA
        vs
Qwen3.5-4B + Answer LoRA
```

如果资源允许，再做 Decision + Answer 两个 LoRA 的组合实验。

## 9. 当前已知限制

- Civil #007 仍可能检索不到第一千一百六十五条/第一千一百八十四条；这不是第一轮 LoRA 的 blocker，但必须保留在 evaluation。
- Retrieval benchmark 样本较少，不能据此宣称真实生产检索率。
- Citation verification 目前是 grounding 基础检查，尚未做到完整法律语义 entailment。
- PDF 全量完整性仍需后续专项审计。

## 10. 下一 AI 接手点

开训后首先检查：

1. accepted/rejected 数量与 domain 分布；
2. Decision 数据是否包含 unknown/no-tool 样本；
3. Answer 数据是否不存在空答案；
4. train/validation/test 是否存在问题泄漏；
5. Raw vs LoRA 是否使用完全一致的 evaluation benchmark 和 Runtime 配置。

不要把训练集 loss 下降当成蒸馏成功。最终标准是 LoRA 相对于 Raw 是否真正改善 Agent 行为，并且没有明显增加错误路由、幻觉 citation 或无关工具调用。
