# Phase 6 Retrieval 诊断与交接报告

> 分支：`fix/phase6`。下一个 AI/开发者应先阅读本文，再继续修改；在 Retrieval 验收完成前不要重新训练 LoRA。

## 1. 当前结论

当前 Retrieval 的主 baseline 是 `GTE-large-zh`。原始 Rewrite（直接替换 query）会伤害部分 case，因此已经改为 additive candidate expansion；Reranker 在当前实验中整体为负，默认关闭。

最新关键发现：#007“楼上邻居漏水把我家天花板泡了”对应的 Gold 法条在 candidate pool 中都没有出现，因此 **Reranker 无法解决 #007**。问题已经被定性为 recall/index/query-representation 层，而不是 ranking 层。

用户最近实验给出的代表性结果：

```text
纯 GTE       Precision@5 ≈ 0.171  Recall@5 ≈ 0.786  MRR ≈ 0.452
旧 Rewrite   Precision@5 ≈ 0.143  Recall@5 ≈ 0.643  MRR ≈ 0.381
```

GTE 对 #005 买卖合同、#006 借款均已有命中；#007 仍然是核心 hard case。

## 2. 已完成

### Router

- `unknown`/abstention；
- labor/civil hard rules；
- loan/debt/delivery/neighbor leakage 等 Civil 规则；
- 条件 Graph 路由；
- 基础 Router benchmark 能力。

### Knowledge ingestion / index

- whole-PDF article-aware splitting；
- 跨页法条不因 page boundary 直接截断；
- 相邻 `第X条` 独立切分；
- normalized embeddings；
- Chroma 显式 cosine metric；
- 索引记录 embedding model 与 normalization 元数据；
- GTE 独立 Chroma A/B 索引；
- chapter/topic enrichment 已接入 embedding text；
- 本阶段新增 article-level legal concept aliases。

### Retrieval

- GTE-large-zh baseline；
- candidate pool；
- `distance` / diagnostic `score` / `rerank_score`；
- optional hybrid；
- optional CrossEncoder；
- Civil query rewrite；
- additive original+rewrite candidate union：rewrite 不再有权删除原始 query 已找到的 candidate；
- Retrieval benchmark 与 candidate-pool diagnostics。

### Citation / Verification

- `retrieved_documents` 与 `citations` 分离；
- citation 必须来自 retrieval evidence；
- answer 中出现的法条编号必须可由 evidence 支撑；
- non-legal 不进入法律检索；
- 失败/空 answer 不进入正式 SFT。

### Phase 5 数据

- Decision SFT 与 Answer SFT 拆分；
- 旧 trajectory 不直接作为最终训练集；
- failure trajectory quarantine；
- trajectory v2 必须等 retrieval 达标后重新生成。

## 3. Retrieval 实验结论

### Top-K

Top-10 → Top-20 的 Recall/MRR 不再提升，继续扩大 K 不是核心解决方案。

### Embedding A/B

GTE-large-zh 明显优于旧 text2vec baseline，并解决 #005 买卖合同 case；BGE-M3 在当前 benchmark 没有显示优势。

### Rewrite

旧实现：`query -> rewrite(query) -> 单次 retrieval`，整体为负。

当前实现：

```text
original query ─┐
                ├─> candidate union -> 同 chunk 取最大 score -> ranking
rewritten query ┘
```

这避免了 rewrite 语义偏移把原始命中删除。

### Reranker

真实实验显示当前 CrossEncoder reranker 整体伤害 GTE，因此默认关闭。

```text
GTE baseline       MRR ≈ 0.405, Recall ≈ 0.714
GTE + reranker     MRR ≈ 0.179, Recall ≈ 0.429
```

更重要的是，#007 的 Gold 没有进入 candidate pool 时，任何 reranker 都不可能把它“召回”。

## 4. 当前 hard case：#007 邻居漏水

问题：

```text
楼上邻居漏水把我家天花板泡了，可以要求赔偿吗？
```

Gold：

```text
第一千一百六十五条
第一千一百八十四条
```

candidate pool diagnostics 的结论：**Gold 不在 Reranker candidate pool 中**。

因此正式定性：

```text
#007
  ↓
Top-K candidate recall 失败
  ↓
Reranker 无法解决
```

当前检索结果虽然围绕“侵权/建筑物/相邻关系”主题，但没有精确到一般侵权责任和财产损失赔偿条款。

### 已做的下一层修复

新增 `knowledge/legal_concepts.py`，为可审计的关键 Civil 条款增加 doctrinal retrieval aliases，例如：

```text
1165 → 一般侵权责任 / 过错责任 / 侵权行为 / 损害 / 赔偿责任
1184 → 财产损失 / 财产损害赔偿 / 损失计算 / 侵权赔偿
288  → 相邻关系 / 用水排水 / 不动产相邻关系
```

这些只是索引增强词，不参与答案生成，也不使用 benchmark question 作为标注。

`knowledge/legal_chunker.py` 会把这些概念追加到 embedding-oriented retrieval text，并把 `legal_concepts` 写入 metadata，同时保留 `original_text`。

**必须重新 ingest** 才能让这批新的向量生效。

## 5. 下一步必须做的真实实验

### P0：重建 GTE + concept-enriched index

```powershell
uv run python -m knowledge.ingest --reset --embedding-model "thenlper/gte-large-zh" --chroma-dir "knowledge/chroma_gte_large_zh"
```

### P0：重新跑 candidate-pool diagnostics

```powershell
uv run python -m evaluation.retrieval_diagnostics data/evaluation/retrieval_benchmark_v2.jsonl --candidate-k 50 --embedding-model "thenlper/gte-large-zh" --chroma-dir "knowledge/chroma_gte_large_zh" --output data/evaluation/candidate_pool_concept_v2.json
```

重点看：

```text
#005 gold_in_candidate_pool
#006 gold_in_candidate_pool
#007 gold_in_candidate_pool
```

### P1：重新跑 baseline

确认 concept enrichment 有没有损伤 #001~#006：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --embedding-model "thenlper/gte-large-zh" --chroma-dir "knowledge/chroma_gte_large_zh"
```

### P1：再跑 additive rewrite

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rewrite --embedding-model "thenlper/gte-large-zh" --chroma-dir "knowledge/chroma_gte_large_zh"
```

## 6. 决策树

```text
#007 Gold 不在 Top-50
    ↓
继续优化 index / query representation

#007 Gold 在 Top-50，但不在 Top-5
    ↓
才值得重新研究 rerank / hybrid ranking

#007 Gold 进入 Top-5
    ↓
Retrieval P0 完成，进入 trajectory v2 验收
```

## 7. 尚未完成

1. concept-enriched GTE index 的真实 A/B 尚未跑完；
2. #007 是否因此进入 Top-50/Top-5 尚未证明；
3. Reranker 仍未作为默认方案；
4. Router benchmark 仍需扩大边界和 non-legal 样本；
5. Retrieval threshold 尚未校准；
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
- 不要针对 #007 写“问题文本 -> 第1165/1184条”的硬编码规则；
- 不要用 LLM 自动补全文法条并回写知识库；
- 不要让 rewrite 直接替换用户原 query；
- 不要在 candidate pool miss 时把问题归因给 reranker。

## 9. 下一 AI 最短路径

```text
1. pytest -q
2. GTE concept-enriched ingest --reset
3. candidate-pool diagnostics
4. baseline retrieval
5. additive rewrite retrieval
6. 分析 #005/#006/#007
7. 若 #007 仍 Top-50 miss → 扩展可审计 Civil ontology / scenario metadata
8. 若 #007 Top-50 hit → 再研究 rerank
9. Retrieval 达标后重新生成 trajectory v2
10. filter/quarantine
11. 人工审核 20~50 条
12. prepare_phase5_data
13. Decision/Answer SFT
14. Raw vs LoRA
```

## 10. 分支与数据

工作分支：`fix/phase6`。

`master` 保持稳定；LoRA 大文件不提交。历史 Raw/LoRA JSON、benchmark、trajectory seed 等数据继续保留。

本报告必须持续更新真实实验数字；不要把“代码支持”写成“实验已经证明”。
