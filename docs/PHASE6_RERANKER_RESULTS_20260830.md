# Phase 6：Reranker 实验结论与当前检索基线

> 分支：`fix/phase6`
>
> 本文记录 2026-08-30 的真实 A/B 结果，供下一 AI 直接接续。没有本地实验数字时，不得把“代码支持”写成“实验已证明”。

## 1. 已完成的真实实验

当前 GTE-large-zh 已作为主 baseline。用户对同一 Retrieval benchmark 做了 GTE、Reranker、Rewrite、Rewrite+Reranker 对比。

用户报告的关键结果：

```text
GTE baseline:
MRR ≈ 0.405
Recall ≈ 0.714

GTE + Reranker:
MRR ≈ 0.179
Recall ≈ 0.429

GTE + Rewrite + Reranker:
MRR ≈ 0.279
```

Rewrite 相对于原始 baseline 已被验证对借款类问题有明显帮助；Reranker 单独加入后整体指标显著下降。

## 2. 当前正式检索策略

生产/trajectory 默认策略固定为：

```text
GTE-large-zh
    + domain-aware query rewrite
    + 不默认启用 CrossEncoder rerank
```

配置：

```text
AGENT_DISTILL_EMBEDDING_MODEL=thenlper/gte-large-zh
AGENT_DISTILL_RETRIEVAL_QUERY_REWRITE=1
AGENT_DISTILL_RETRIEVAL_RERANK=0
```

Reranker 仅保留为实验选项，不作为默认生产路径。

## 3. 为什么不默认 Reranker

用户实验显示 Reranker 在当前 corpus/query/模型组合下会把一些正确的实体性法律条款降权，把程序性条款推到前面。

因此当前不能笼统声称“CrossEncoder 一定优于 embedding”。它是二阶段模型，效果高度依赖候选文本、query、训练分布和法律语料域。

## 4. #007 邻居漏水

当前仍是核心 hard case：

```text
楼上邻居漏水把我家天花板泡了，可以要求赔偿吗？
Gold：民法典 1165、1184
```

所有当前组合仍不能稳定召回这些 gold，因此现阶段更应视为**候选召回/索引表示问题**，而不是继续调 Top-K 或 Reranker 权重。

本阶段新增了 chapter-aware topic enrichment：

```text
第七章 侵权责任
→ 一般侵权 / 过错责任 / 损害赔偿 / 财产损害 / 人身损害 / 相邻关系
```

这些主题词会进入 embedding-oriented document text，但不硬编码 benchmark gold article number，避免数据泄漏。

重新 ingest 后才生效。

## 5. #005 / #006 回归状态

### #005 买卖合同逾期交货

GTE 已解决到候选集合中；该样本不应在后续优化中退化。

### #006 借款到期不还

Rewrite 已证明有帮助。后续任何索引/embedding 改动都必须重新验证此样本。

## 6. 本阶段已经新增代码

- `knowledge/topic_enrichment.py`
- `knowledge/legal_chunker.py`：embedding 文本加入章节/主题增强
- `configs/settings.py`：GTE 默认、Rewrite 默认开启、Reranker 默认关闭
- `evaluation/retrieval_benchmark.py`：支持 embedding/chroma 实验参数
- `evaluation/reranker_diagnostics.py`：检查 reranker 是否真实加载

## 7. 下一步必须做

### P0：重建 GTE 索引

主题增强修改后必须重新：

```powershell
uv run python -m knowledge.ingest --reset --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh
```

### P0：重新跑 GTE baseline + rewrite

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh

uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rewrite --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh
```

重点看 #005/#006/#007。

### P0：确认 #007 是否被主题增强召回

如果 #007 仍 Top-20 miss，则继续进入“法律主题标签 / article scenario index”路线，而不是继续调 reranker。

### P1：Reranker 仅保留诊断用途

只有在候选 Top-20 已包含 gold 的样本上，才讨论 reranker 的排序能力；如果 gold 不在候选集，reranker 没有能力把它“凭空找出来”。

### P1：Citation semantic grounding

当前已能检查引用来自 retrieval，但还未证明引用内容与回答中的法律推理正确对应。下一阶段增加 entailment/人工审核。

### P1：Trajectory v2

只有新的 GTE+Rewrite 检索结果稳定后才重新生成 20~50 条 trajectory v2，并隔离所有空答案/verification fail 样本。

## 8. 明确不要做

- 不要把 Reranker 设为默认；
- 不要继续无依据增加 Top-K；
- 不要把 diagnostic score 当概率；
- 不要重新训练 LoRA，直到 Retrieval + Citation + Verification 通过验收；
- 不要把 #007 做成 article-number hardcode；
- 不要把模型 benchmark 的通用成绩当作本项目法律检索证明。

## 9. 当前阶段一句话

> **GTE-large-zh + Rewrite 是当前最佳候选生产基线；Reranker 已被真实 A/B 证明在当前配置下有害，因此降级为实验工具；#007 仍需要索引表示/法律场景增强来解决。**
