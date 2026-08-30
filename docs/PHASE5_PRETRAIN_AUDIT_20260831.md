# Phase 5 LoRA 开训前审计记录（2026-08-31）

## 当前结论

本阶段可以进入第一轮 Qwen3.5-4B LoRA 实验。Retrieval 冻结为可解释 baseline，不再因单个 hard case 阻塞训练。

### 冻结配置

```text
Embedding: GTE-large-zh
Reranker: OFF
Hybrid: OFF
Rewrite: 第一轮训练使用 OFF
Index: normalized embeddings + cosine
```

## 训练数据状态

最近一次数据准备：

```text
有效 trajectory: 162
过滤失败/空答案/非法路由: 0
Decision SFT: 162
Answer SFT: 162
```

注意：这些数字表示通过结构性过滤，不表示 162 条全部经过人工法律语义审计。

AI 抽检（用户反馈）发现：

- 总体质量约 90% 可用；
- 检索与 citation 总体明显优于历史轨迹；
- 仍有重复法条伪 chunk、少量路由误判、少量答案截断。

## 本次已修复

1. PDF / Retriever 重复法条：过滤只有“第X条”标题、没有正文的抽取伪 chunk。修复位置为 `knowledge/legal_chunker.py`，不在 Retriever 输出端简单去重。
2. Router：新增/强化家具、网购、物业、绿地、试用期、辞职、离职等边界案例规则。
3. Negative routing：unknown/non-legal + no tool 仍应作为 Decision SFT 负路由监督保留。
4. SFT audit：支持最终 `messages` 格式，避免把正常 ChatML 数据误判为空。

## 未修复/有意延期

### P1

- 少量 Teacher answer 截断：后续可增加 finish_reason/截断检测，并在生成侧重试；当前不阻塞第一轮 LoRA。
- Citation semantic entailment：当前为 grounding 基础校验，尚未完成“每个法律主张是否被对应法条充分支持”的完整语义审计。

### P2

- Civil #007 邻居漏水仍是 hard recall case：Gold 1165/1184 曾未进入 candidate pool。已保留为 evaluation hard case，不针对单题硬编码生产规则。
- PDF 全量法条完整性专项审计尚未完成。

## 开训前必须做

1. `uv run pytest -q` 全绿。
2. 审计 Decision / Answer 的 domain 分布、重复问题、空答案。
3. 从 `agent_trajectory_v2_accepted.jsonl` 随机抽样检查 legal / civil / labor / unknown。
4. 确认训练/验证/测试没有 query 泄漏。
5. 第一轮 LoRA 保持固定超参数，不同时修改 Retrieval。

## 第一轮 LoRA 实验

```text
Qwen3.5-4B Raw
vs
Qwen3.5-4B + Decision LoRA
vs
Qwen3.5-4B + Answer LoRA
```

核心验收不是 train loss，而是：

- Router/domain accuracy
- Tool selection accuracy
- Answer quality
- Citation grounding
- Verification validity
- Task success

## 交接原则

不要因为 `accepted=162` 就声称数据是 gold。不要把历史 `agent_trajectory.jsonl` 直接用于最终 SFT。#007 仍需保留为 hard case。LoRA 权重不提交 Git。
