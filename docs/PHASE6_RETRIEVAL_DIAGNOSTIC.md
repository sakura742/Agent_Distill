# Phase 6 Retrieval 诊断与交接报告

> 分支：`fix/phase6`。下一个 AI/开发者应先阅读本文，再继续修改；在 Retrieval 验收完成前不要重新训练 LoRA。

## 1. 当前结论

最新真实实验表明，`GTE-large-zh` 是当前检索 baseline。Rewrite 在原实现中采用“替换原查询”的方式，会伤害 #005 买卖合同，同时帮助 #006 借款；因此不能把 Rewrite 直接作为默认查询。

用户报告的当前对比：

```text
纯 GTE
Precision@5 ≈ 0.171
Recall@5    ≈ 0.786
MRR         ≈ 0.452

GTE + Rewrite（旧：替换查询）
Precision@5 ≈ 0.143
Recall@5    ≈ 0.643
MRR         ≈ 0.381
```

逐样本：

- #001 加班：Rewrite 不变；
- #002 拖欠工资：Rewrite 不变；
- #003 违法辞退：Rewrite 不变；
- #004 社保：Rewrite 不变；
- #005 买卖合同：纯 GTE 命中且排第 1，旧 Rewrite 造成 miss；
- #006 借款：Rewrite 将正确法条提高到第 1；
- #007 邻居漏水：所有组合仍 miss。

因此最新策略是：

```text
GTE-large-zh = baseline
Rewrite       = additive candidate expansion，不再替换原查询
Reranker      = experimental only，不进入默认链路
#007          = 仍是 recall/index hard case
```

## 2. 本轮已完成

### Router

- `unknown`/abstention；
- labor/civil hard rules；
- loan / debt / delivery / neighbor leakage 等 Civil 规则；
- 条件 Graph 路由；
- 基础 Router benchmark 能力。

### Knowledge ingestion

- whole-PDF article-aware splitting；
- 跨页法条不因 page boundary 直接截断；
- 相邻 `第X条` 独立切分；
- normalized embeddings；
- Chroma 显式 cosine metric；
- 索引记录 embedding model 与 normalization 元数据。

### Retrieval

- GTE-large-zh baseline；
- 独立 embedding/chroma A/B；
- candidate pool；
- `distance` / diagnostic `score` / `rerank_score`；
- optional hybrid；
- optional CrossEncoder；
- Civil query rewrite；
- **本轮将 rewrite 从“替换 query”改成“原 query + 改写 query 双路召回后 union，按同一 chunk 的最大 embedding score 保留”**。

这样可以避免 #005 这类原查询已经能准确召回的 case 因 rewrite 偏移而丢失，同时允许 #006 从改写查询获得额外候选。

### Citation / Verification

- `retrieved_documents` 与 `citations` 分离；
- citation 必须来自 retrieval evidence；
- answer 中的法条编号必须与证据一致；
- non-legal 不进入法律检索；
- 失败/空 answer 不进入正式 SFT。

### Phase 5 数据

- Decision SFT 与 Answer SFT 拆分；
- 旧 trajectory 不直接作为最终训练集；
- failure trajectory quarantine；
- trajectory v2 必须等 retrieval 达标后重新生成。

## 3. Retrieval 实验结论

### Top-K

此前实验：

```text
Top-5  Recall ≈ 0.571 / MRR ≈ 0.298
Top-10 Recall ≈ 0.714 / MRR ≈ 0.313
Top-20 Recall ≈ 0.714 / MRR ≈ 0.313
```

Top-10 → Top-20 无进一步收益，因此继续扩大 K 不是解决方案。

### Embedding A/B

GTE-large-zh 明显优于旧 text2vec baseline，并解决 #005 买卖合同 case；BGE-M3 在当前 benchmark 没有更好表现。

### Rewrite

旧实现：

```text
query -> rewrite(query) -> 单次 retrieval
```

结果对整体有负贡献，且 #005 从 hit 变成 miss。

因此新实现：

```text
original query ─┐
                ├─> candidate union -> best score per chunk -> ranking
rewritten query ┘
```

Rewrite 不再有权删除原查询已经找到的 candidate。

### Reranker

真实实验表明当前 CrossEncoder reranker 在整体上显著伤害 GTE：

```text
GTE baseline       MRR ≈ 0.405, Recall ≈ 0.714
GTE + reranker     MRR ≈ 0.179, Recall ≈ 0.429
```

因此 reranker 目前不是默认方案。只有后续在更强 candidate pool、领域化或加权融合后证明有效，才重新考虑进入正式链路。

## 4. 当前 hard cases

### #005 买卖合同逾期交货

GTE 已经可以命中 `第五百七十七条`，所以不能再让 rewrite 破坏这个 case。后续以 regression test 保护。

### #006 借款到期不还

Rewrite 有帮助，说明口语 query 到法律概念之间存在表达 gap。新的 additive fusion 应保持原始命中，同时利用 rewrite candidate。

### #007 邻居漏水

多个 embedding / rewrite / reranker 组合均未稳定召回 `第一千一百六十五条` / `第一千一百八十四条`。

这说明当前主要矛盾仍是 recall/index representation，而不是简单 ranking。

下一步优先：

```text
法律主题/场景 metadata
+
语义增强的法条索引文本
+
必要时人工构造高质量 Civil hard-case query variants
```

不能写死 `#007 -> 1165/1184` 这种 benchmark-specific 规则到生产 retriever。

## 5. Embedding/index contract

查询模型与 Chroma 必须完全一致：

- 同一 `embedding_model`；
- 同一 normalization；
- 同一 `hnsw:space=cosine`；
- 同一 collection；
- 不同 embedding 使用不同 `chroma_db_dir`。

当前 Retriever 会提前检查模型身份与维度，避免 Chroma 在 query 时才抛 768 vs 1024 之类的底层错误。

## 6. Citation / verification 剩余问题

当前 grounding 基础校验已经存在，但还没有完成真正的 semantic entailment：

```text
citation 是否与回答中的具体主张一致？
```

不能因为 `第X条` 出现在 answer 和 evidence 两边，就直接认为法律解释一定正确。

## 7. 尚未完成

1. GTE + additive rewrite 的完整 benchmark 尚未由用户重新跑出最终数字；
2. #007 的 topic/scene index enhancement 尚未完成最终 A/B；
3. Router benchmark 仍需扩充边界样本；
4. Retrieval threshold 尚未校准；
5. Reranker domain-specific / score fusion 尚未继续验证；
6. Citation semantic entailment 尚未完成；
7. PDF 全量完整性审计尚未完成；
8. trajectory v2 尚未经过 20~50 条人工抽样验收；
9. 新 Decision/Answer LoRA 尚未训练；
10. Raw vs LoRA 新一轮 benchmark 尚未开始。

## 8. 明确禁止

- 不要把旧 `agent_trajectory.jsonl` 直接用于最终 LoRA；
- 不要因为 Top-K 增大而声称 retrieval 已修复；
- 不要把 diagnostic score 当概率；
- 不要在没有 A/B 的情况下宣称 reranker/hybrid/rewrite 有效；
- 不要针对 #007 写硬编码特殊规则；
- 不要用 LLM 自动补全文法条并回写知识库；
- 不要让 rewrite 直接替换用户原 query；必须保持 original retrieval candidate。

## 9. 下一 AI 最短路径

```text
1. pytest -q
2. 重建 GTE index（如 index 未包含最新 topic enrichment）
3. baseline GTE
4. GTE + additive rewrite
5. 比较 #005/#006/#007
6. 如果 #007 仍 Top-20 miss -> topic metadata / index enhancement
7. Retrieval 达标后重新生成 trajectory v2
8. filter/quarantine 失败样本
9. 人工审核 20~50 条
10. prepare_phase5_data
11. Decision/Answer SFT
12. Raw vs LoRA
```

## 10. 分支与数据

工作分支：`fix/phase6`。

`master` 保持稳定；LoRA 大文件不提交。历史 Raw/LoRA JSON、benchmark、trajectory seed 等数据继续保留。

本报告必须持续更新真实实验数字；不要把“代码支持”写成“实验已经证明”。
