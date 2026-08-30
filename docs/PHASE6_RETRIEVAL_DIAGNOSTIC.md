# Phase 6 Retrieval 诊断、收尾与交接报告

> 状态：**Phase 6 收尾，不再继续迭代 Retrieval**。
>
> 工作分支：`fix/phase6`。本报告记录最终成果、真实实验结论、已知限制，以及最后阻塞项目继续推进的环境问题。

## 1. 最终决策

本项目在当前阶段正式冻结 Retrieval，不再继续为了单个 hard case 无限调参。

当前推荐 baseline：

```text
Embedding: GTE-large-zh
Reranker: OFF
Hybrid: OFF
Rewrite: 不作为第一轮 LoRA 的固定依赖；仅保留为可选 A/B 能力
```

Retrieval 已经完成足够的工程化与诊断工作，可以为第一轮 LoRA 实验提供稳定基线。

但是由于后期 Windows 环境出现持续性的 `WinError 6714 (ERROR_CURRENT_TRANSACTION_NOT_VALID)`，完整测试与 LoRA 实训未能在当前环境最终验收，因此本项目以**“LoRA pipeline 已实现、训练前数据已准备，但未完成最终训练/评测”**收尾，不宣称 LoRA 已成功。

## 2. 已完成工作

### Router

- labor / civil / unknown 三路由；
- non-legal abstention；
- labor/civil 关键词规则；
- 借款、借钱、借条、买卖、逾期交货、邻居漏水、天花板等 Civil 边界词；
- Graph 条件路由；
- Router benchmark；
- 修复家具尺寸不符、物业改绿地、试用期辞退等 unknown 误判案例。

### Knowledge / Index

- PDF 按法律条款粒度拆分；
- 跨页条款处理；
- 避免相邻文章合并；
- 清理只有“第X条”标题、没有正文的伪 chunk；
- normalized embedding；
- Chroma cosine metric；
- 索引记录 embedding model / normalization metadata；
- embedding-index dimension / identity contract；
- GTE-large-zh 独立索引；
- chapter/topic enrichment；
- article-level legal concept enrichment；
- 保留 `original_text`，增强词只用于 retrieval。

### Retrieval

- `Precision@K` / `Recall@K` / `MRR`；
- Top-5 / Top-10 / Top-20 实验；
- candidate-pool diagnostics；
- score / distance / rerank_score 分离；
- optional hybrid；
- optional CrossEncoder reranker；
- query rewrite；
- additive query candidate union，避免 rewrite 删除原始命中；
- embedding A/B：text2vec vs BGE-M3 vs GTE-large-zh。

### Citation / Verification

- `retrieved_documents` 与 `citations` 分离；
- citations 只能来源于检索证据；
- answer 中出现的法条编号需要能和 evidence 对齐；
- non-legal 不进入法律检索；
- 空答案 / verification failed / 非法 tool 路由进入 quarantine。

### Phase 5 数据

- Decision SFT / Answer SFT 分离；
- unknown/no-tool 负路由监督保留；
- failure trajectory quarantine；
- ChatML SFT 数据审计器；
- trajectory v2 重新生成链路；
- 训练数据已成功生成：Decision 162、Answer 162。

## 3. 真实实验结论

### Top-K

Top-10 → Top-20 后 Recall / MRR 基本不再增长，因此继续扩大 K 不是主要解法。

### Embedding

GTE-large-zh 在当前法律 benchmark 中明显优于旧 text2vec baseline，并成功解决：

- #005 买卖合同逾期交货；
- #006 借款到期拒绝还款。

BGE-M3 在当前 benchmark 上没有表现出优势，因此停止继续无目的更换 embedding。

### Rewrite

旧的“直接替换 query”策略会伤害 #005；后续改为 original + rewritten candidate union，以避免语义偏移删除原始命中。

用户实际实验中曾观察到纯 GTE：

```text
Precision@5 ≈ 0.171
Recall@5    ≈ 0.786
MRR         ≈ 0.452
```

旧 Rewrite 曾降低到：

```text
Precision@5 ≈ 0.143
Recall@5    ≈ 0.643
MRR         ≈ 0.381
```

因此第一轮 LoRA 默认采用纯 GTE 更适合作为干净 baseline。

### Reranker

真实实验显示当前 CrossEncoder reranker 整体为负：

```text
GTE baseline       MRR ≈ 0.405, Recall ≈ 0.714
GTE + reranker     MRR ≈ 0.179, Recall ≈ 0.429
```

因此 Reranker 默认关闭，仅保留实验能力。

## 4. Hard case：#007 邻居漏水

问题：

```text
楼上邻居漏水把我家天花板泡了，可以要求赔偿吗？
```

Gold：

```text
第一千一百六十五条
第一千一百八十四条
```

candidate-pool diagnostics 证明：Gold 未进入 candidate pool。

因此可以确定：

```text
#007
→ candidate recall 失败
→ 不是 reranking 问题
→ Reranker 无法“凭空召回” Gold
```

进一步增加章节/topic 与 article-level legal concept enrichment 后，代码已经具备真实实验条件，但由于项目最终收尾，没有继续把 #007 作为 blocker 无限迭代。

#007 应继续作为 evaluation / hard case 保留，而不是构造“问题文本 → 指定法条”的硬编码映射。

## 5. 训练前数据状态

最终 trajectory v2 数据通过程序性门禁后：

```text
Accepted trajectory: 162
Rejected: 0
Decision SFT: 162
Answer SFT: 162
```

这里的含义是：162 条通过了自动结构过滤；**并不意味着 162 条全部经过人工/LLM 法律语义审核**。

AI 抽检过的一批样本总体质量明显优于旧版本，主要风险已收敛为：

- 少量检索重复 / PDF chunk 冗余；
- 少量 answer 截断；
- 少量 Router unknown 误判；
- Citation semantic entailment 尚未完全自动化。

## 6. LoRA 阶段现状

Phase 5 训练脚本已具备：

- Qwen3.5-4B；
- 4-bit quantization；
- LoRA；
- Decision mode；
- Answer mode；
- 独立输出目录；
- backward-compatible trajectory serializer。

训练数据已经准备完成。

但在实际执行：

```powershell
uv run python -m distill.train_phase5 --mode decision
uv run python -m distill.train_phase5 --mode answer
```

之前，项目环境出现持续的 Windows：

```text
WinError 6714
ERROR_CURRENT_TRANSACTION_NOT_VALID
```

该错误曾在：

- `importlib` 枚举 `.venv/site-packages`；
- 项目 `distill` 目录；
- pytest cache；
- 临时目录；

等多个无关路径出现，因此判断为 Windows / 文件系统事务层环境问题，而不是项目业务逻辑问题。

即使切换项目目录并尝试系统层修复后，问题仍复现。

**因此本项目不再继续消耗时间处理该环境问题。**

## 7. 尚未完成

1. Qwen3.5-4B Decision LoRA 尚未正式完成训练；
2. Qwen3.5-4B Answer LoRA 尚未正式完成训练；
3. Raw vs Decision LoRA vs Answer LoRA 尚未完成统一 benchmark；
4. #007 concept-enriched index 的最终收益没有完成最终闭环验收；
5. Citation semantic entailment 尚未达到完整法律级验证；
6. PDF 全量完整性审计没有最终完成；
7. 训练/validation/test 的严格问题去重切分尚未形成最终报告；
8. Windows `WinError 6714` 环境问题仍未从根本上解决。

## 8. 为什么此处收尾是合理的

到此已经完成了一个完整的工程研究闭环：

```text
RAG / Router 初版问题
        ↓
真实轨迹抽检
        ↓
发现 Router / Citation / Verification / Retrieval 问题
        ↓
分层 benchmark
        ↓
Top-K 排查
        ↓
Embedding A/B
        ↓
GTE 胜出
        ↓
Rewrite A/B
        ↓
发现旧 Rewrite 有副作用
        ↓
Reranker A/B
        ↓
证明当前 reranker 为负
        ↓
candidate-pool diagnostics
        ↓
证明 #007 属于 recall/index representation hard case
        ↓
Legal concept enrichment
        ↓
Trajectory v2
        ↓
162 条 Decision / Answer SFT
        ↓
准备进入 LoRA
        ↓
Windows 环境层阻塞
        ↓
项目收尾
```

因此现在已经有足够完整的工程成果和实验结论，不必为了“100% 完成 LoRA”继续无限修复环境或 Retrieval 长尾。

## 9. 下一位 AI / 开发者如果继续

最短路径：

```text
1. 先确认新的 Windows / Linux / WSL 环境稳定
2. 不修改冻结 Retrieval baseline
3. 使用已生成的 phase5_decision.jsonl / phase5_answer.jsonl
4. 建议重新做严格 train/validation/test 去重切分
5. 运行 Decision LoRA
6. 运行 Answer LoRA
7. 使用相同 benchmark 比较 Raw / Decision LoRA / Answer LoRA
8. 最后再决定是否继续处理 #007
```

不要因为 LoRA loss 下降就声称蒸馏成功。真正的成功标准仍然是：

- Router Accuracy；
- Tool Selection Accuracy；
- Answer Quality；
- Citation Grounding；
- Verification Validity；
- Task Success Rate；

相对于 Qwen3.5-4B Raw 的真实提升。

## 10. Git / 数据原则

- 当前工作分支：`fix/phase6`；
- `master` 用于最终稳定版本；
- 不提交 LoRA 权重和其他大模型文件；
- benchmark / trajectory seed / 小型 JSON 评估结果应保留；
- 所有实验结果必须注明“代码支持”和“真实跑过的实验”之间的区别。

## 11. 最终状态

```text
Phase 1         ✓
Phase 2         ✓
Phase 3         ✓
Phase 4         ✓
Phase 5 pipeline ✓
Phase 6         ✓ 诊断与冻结
LoRA training   ⏸ 环境阻塞，未最终验收
```
