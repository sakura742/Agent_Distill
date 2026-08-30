# Phase 6 Reranker 阶段交接

## 当前基线

根据用户本地 A/B 实验：GTE-large-zh 相比原 embedding baseline 明显更好；其中买卖合同逾期交货（#005）已从 Top-20 miss 变为命中，MRR 明显提升。BGE-M3 在当前 benchmark 上没有取得更好结果，因此当前 Retrieval baseline 暂定为 `thenlper/gte-large-zh`。

但是，GTE 的 superiority 仍然只在当前小型法律 benchmark 上成立，不能外推到一般法律语料。

## 当前硬问题：#007 邻居漏水

`楼上邻居漏水把我家天花板泡了，可以要求赔偿吗？` 在当前 embedding 配置中仍是 hard case。它需要区分：

- 一般侵权责任；
- 相邻关系；
- 特殊侵权/建筑物相关责任；
- 财产损害赔偿。

下一实验优先使用 candidate Top-20 + CrossEncoder rerank，看 gold 是否已经在候选集中。

## 重要实验卫生规则

### 1. Embedding 与索引必须一一对应

不能使用 GTE query embedding 查询 text2vec/BGE 生成的 Chroma index。

当前 ingest 会在 collection metadata 写入：

```text
embedding_model
hnsw:space=cosine
```

Retriever 还会检查模型名称、metric 和向量维度。

### 2. Reranker 失败不能静默 fallback

请求 `rerank=True` 时，如果没有 `AGENT_DISTILL_RERANKER_MODEL`，或者 CrossEncoder 加载/推理失败，必须直接失败并说明原因。否则 benchmark 会把“实际上没有 rerank”误当成 rerank 结果。

### 3. Reranker 只对候选集负责

如果 #007 的 gold 根本不在 candidate Top-20，那么 reranker 不可能解决问题；此时应转向 query expansion、索引文本增强或 embedding/indexing。

## 推荐执行顺序

1. 确认 GTE 索引是重新 ingest 得到的：

```powershell
uv run python -m knowledge.ingest --reset --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh
```

2. 配置本地 CrossEncoder，例如：

```powershell
$env:AGENT_DISTILL_RERANKER_MODEL="D:\py\embeddings\bge-reranker-v2-m3"
```

3. 先做 reranker diagnostics：

```powershell
uv run python -m evaluation.reranker_diagnostics data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --candidate-k 20 --output data/evaluation/reranker_diagnostics.json
```

必须确认：

```text
reranker_model_configured = true
rerank_scores 不为 null
```

4. 再做 Retrieval A/B：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh --output data/evaluation/gte_baseline_top5.json
```

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rerank --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh --output data/evaluation/gte_rerank_top5.json
```

5. 对 #005/#006/#007 单独检查 Top-20 candidate 与 Top-5 reranked 排名。

## 结果解释规则

- gold Top-20 有 + rerank Top-5 进来：ranking 问题，reranker 有价值。
- gold Top-20 有 + rerank 仍不进 Top-5：当前 CrossEncoder 不适合或候选/输入表达仍有问题。
- gold Top-20 都没有：retrieval recall 问题，继续优化 query/index/embedding，不要归咎于 reranker。

## 当前未完成

- Reranker 本机实际 A/B；
- #007 是否进入 GTE Top-20 的最终确认；
- Civil topic/scenario metadata；
- 更大规模 Router benchmark；
- trajectory v2；
- Citation semantic entailment；
- Decision/Answer LoRA 重训。

## LoRA 训练门槛

在上述 Retrieval 问题处理完成、trajectory v2 经过人工抽样前，继续禁止训练新的 LoRA。
