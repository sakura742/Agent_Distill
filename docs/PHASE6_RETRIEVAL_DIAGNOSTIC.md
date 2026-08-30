# Phase 6 Retrieval 诊断与交接报告

> 分支：`fix/phase6`。下一个 AI/开发者应先阅读本文，再继续修改；在 Retrieval 验收完成前不要重新训练 LoRA。

## 1. 当前结论

最新真实 Retrieval 实验已经完成 Top-5/10/20 扩大 K 诊断。结果显示 Top-10→Top-20 的 Recall/MRR 不再提升，说明扩大候选数量不能解决两个顽固 Civil query 的问题；因此当前工作重点从“扩大 K”转向“候选召回质量/查询表达/重排”。

用户报告的最新汇总：

```text
Recall@5  ≈ 0.571
Recall@10 ≈ 0.714
Recall@20 ≈ 0.714
MRR@5     ≈ 0.298
MRR@10    ≈ 0.313
MRR@20    ≈ 0.313
```

其中：

- #005 买卖合同逾期交货：gold 第577条到 Top-20 仍未出现；
- #007 邻居漏水：gold 第1165/1184条到 Top-20 仍未出现；
- #006 借款纠纷：Top-5 miss，但更大 K 可以找到，属于更接近 ranking 问题；
- Router 最近独立评估约 86.7%，主要弱项是 Civil 漏报/边界问题。

因此不能继续把 #005/#007 解释成简单的“Top-K 不够”或“纯 rerank 就能解决”。

## 2. 已完成

### Router

- `unknown`/abstention；
- labor/civil hard rules；
- neighbor/leakage/traffic/loan/lease 等 Civil 场景规则；
- 条件 Graph 路由；
- Router benchmark 基础入口。

### Knowledge ingestion

- whole-PDF article-aware splitting；
- 跨页法条不因 page boundary 直接截断；
- 相邻 `第X条` 独立切分；
- normalized embeddings；
- Chroma collection 显式 cosine metric。

### Retrieval

- candidate pool；
- `distance` / diagnostic `score` / `rerank_score`；
- optional hybrid；
- optional CrossEncoder rerank；
- query rewrite 能力；
- 新的 Retrieval benchmark v2。

### Citation / Verification

- `retrieved_documents` 与 `citations` 分离；
- citation 必须来自 retrieval evidence；
- answer 中的法条编号必须与证据一致；
- non-legal 不进入法律检索；
- 失败/空 answer 不进入正式 SFT。

### Phase 5 数据

- Decision SFT 与 Answer SFT 拆分；
- 旧 trajectory 不直接作为最终训练集；
- failure trajectory 可以 quarantine，用于 error analysis。

## 3. 重要历史根因

### 3.1 Query / document embedding contract 曾经不一致

历史代码使用过 `1 - distance / 2` 作为 score，并且 query/document embedding normalization 与 collection metric 不统一，造成 200+ distance 和约 0.004 的诊断 score。

现在统一为 normalized embedding + cosine collection。**必须重新 ingest 才能真正生效。**

### 3.2 `citations` 曾经全量复制 retrieval

旧 trajectory 中 `citations == retrieved_documents`，导致噪声法条直接进入 supervision。

### 3.3 verification 曾经只有存在性检查

旧逻辑只要 answer/citations 存在就可能通过，无法识别：

- answer 引用检索不到的法条；
- citation 与问题无关；
- non-legal 问题却出现法律 evidence。

## 4. Top-K 诊断的意义

Top-5/10/20 实验不是最终优化，而是定位。

如果 gold：

```text
Top-5 miss
Top-10 hit
Top-20 hit
```

优先考虑 ranking/re-ranker。

如果：

```text
Top-20 still miss
```

则问题更可能在：

- embedding 表达能力；
- query formulation；
- index/chunk representation；
- gold 标注与实际法条语义不一致。

当前 #005/#007 属于第二类，因此不能只上 reranker 就认为问题可以解决。

## 5. 本轮新增方向

### 5.1 Query rewrite

新增 `knowledge/query_rewrite.py`，支持：

- 逾期交货 → 合同迟延履行 / 违约责任；
- 借款不还 → 借款期限 / 到期返还 / 债务人；
- 邻居漏水 → 相邻关系 / 侵权责任 / 财产损害赔偿；
- 押金/租房 → 租赁合同 / 押金返还；
- 交通事故 → 侵权责任 / 人身及财产损害。

这是实验能力，不应因为代码存在就默认宣称有效。

### 5.2 Retrieval benchmark A/B

benchmark 现在支持独立比较：

```text
embedding-only
embedding + hybrid
embedding + rerank
embedding + rewrite
```

并记录：

- Precision@K
- Recall@K
- MRR
- distance
- score
- rerank_score

## 6. 当前真正的 P0

### P0-1：重新 ingest

```powershell
uv run python -m knowledge.ingest --reset
```

### P0-2：对同一 gold 集完成以下实验

```powershell
# baseline
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5

# rewrite
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rewrite

# rerank
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rerank

# rewrite + rerank
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rewrite --rerank
```

前提：本地必须配置一个真正可加载的 CrossEncoder，否则 `--rerank` 只能走 no-op fallback。

### P0-3：比较 #005/#006/#007 的结果

重点不是只看平均值，要看三个 case：

```text
#005 买卖合同逾期交货
#006 借款到期不还
#007 邻居漏水
```

如果 rewrite 能救 #005/#007，说明主要是 query formulation；如果仍然 Top-20 miss，则进入 embedding/index A/B。

## 7. Embedding 模型 A/B 原则

不要直接把项目永久切换到某个“听起来更强”的模型。

应保持：

- 同一 PDF；
- 同一 chunker；
- 同一 gold；
- 同一 cosine metric；
- 同一 Top-K；

只替换 `embedding_model_name`，然后重建两个独立 collection，比较 Recall@5/10/20 和 MRR。

这样才能证明改善来自 embedding，而不是其它变量。

## 8. Civil metadata/topic labels

如果 embedding A/B 仍不能稳定召回 Civil gold，再增加 article topic/scenario metadata，例如：

```text
侵权-一般
侵权-特殊
相邻关系
财产损害
合同-买卖
合同-借款
合同-租赁
违约责任
```

但标签必须来自可验证的规则/人工标注，不能让模型自我生成后直接作为 gold。

## 9. 尚未完成

1. Retrieval threshold 尚未完成 benchmark 校准；目前不应把固定 `0.45` 当作真实相关概率。
2. Reranker 尚未完成真实本地模型 A/B。
3. Query rewrite 尚未完成 benchmark A/B。
4. 更强 embedding 模型尚未完成严格 A/B。
5. Router benchmark 需要继续扩展边界和 non-legal 样本。
6. Citation semantic entailment 尚未完成。
7. 原 PDF 全量完整性审计尚未完成。
8. trajectory v2 尚未经过 20~50 条人工抽样验收。
9. 新 Decision/Answer LoRA 尚未训练。
10. Raw vs LoRA 新一轮 benchmark 尚未开始。

## 10. 明确禁止

- 不要把旧 `agent_trajectory.jsonl` 直接用于最终 LoRA；
- 不要因为 Top-K 增大而声称 retrieval 已修复；
- 不要把 diagnostic score 当概率；
- 不要在没有 A/B 的情况下宣称 reranker/hybrid/rewrite 有效；
- 不要为了提高 benchmark 分数针对 #005/#007 写硬编码特殊规则；
- 不要用 LLM 自动“补全文法条”并回写知识库。

## 11. 下一 AI 的最短路径

```text
1. pytest -q
2. ingest --reset
3. baseline / rewrite / rerank / rewrite+rERANK A/B
4. 分析 #005/#006/#007
5. 如果 #005/#007 仍 Top-20 miss → embedding model A/B
6. 再考虑 topic metadata
7. Retrieval 达标后重新生成 trajectory v2
8. 人工审核 20~50 条
9. prepare_phase5_data.py
10. Decision/Answer SFT
11. Raw vs LoRA
```

## 12. 分支与数据

工作分支：`fix/phase6`。

`master` 保持稳定；LoRA 大文件不提交。历史 Raw/LoRA JSON、benchmark、trajectory seed 等数据继续保留。

本报告持续更新；下一次更新必须加入真实 A/B 数字，不要只写“代码支持”。
