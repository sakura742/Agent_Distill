# Phase 6 Retrieval 诊断与交接报告

> 分支：`fix/phase6`。下一个 AI/开发者应先阅读本文，再继续修改；**在 Retrieval 验收完成前不要重新训练 LoRA**。

## 当前结论

最新真实 retrieval benchmark（7 条，embedding-only，Top-5）得到：Precision@5 ≈ 0.143、Recall@5 ≈ 0.571、MRR ≈ 0.333。labor 有部分 gold 命中，3 个 civil query 全部 Top-5 miss。

已经确认一个此前未被发现的上游根因：查询侧和历史建库侧的 embedding normalization/collection metric 曾不一致。历史结果出现 188~276 的 distance 和约 0.004 的诊断 score，说明不能继续把旧 score 当成相关性概率。代码现在统一 normalized embeddings，并显式创建 cosine collection；必须重新 `ingest --reset` 后才可验证修复是否真正生效。

## 已完成

- Router：`unknown`/abstention、条件路由、labor/civil hard rules；测试通过。
- Citation：`retrieved_documents` 与 `citations` 分离；答案引用不能超出检索证据。
- Verification：检查 answer 非空、citation 来源、答案中的法条编号；区分 non-legal。
- 法条切块：whole-PDF article-boundary 切分，避免纯 page-boundary 截断跨页法条；相邻法条按 `第X条` 拆分。
- Retrieval：candidate pool、可选 CrossEncoder、可选 hybrid；结果保留 `distance`、`score`、`rerank_score`。
- Score：移除旧 `1-distance/2`；score 是诊断/排序分数，不是概率。
- Embedding/Collection：ingest 与 query 均 `normalize_embeddings=True`，collection 显式 `hnsw:space=cosine`。
- Phase 5：Decision/Answer SFT 分离；空答案和验证失败样本不会进入正式 SFT。
- Retrieval benchmark v2：canonical `law_name + article` gold reference。
- Router benchmark v2：labor/civil/non-legal/ambiguous gold query。
- Runtime rerank 配置统一，不再把 rerank 写死为开启。

## 真实 Retrieval benchmark

用户本地运行 embedding-only Top-5：

```text
samples = 7
top_k = 5
rerank = false
precision_at_k = 0.142857
recall_at_k = 0.571429
mrr = 0.333333
```

现象：

- labor 加班：命中第44条，排名第3；
- labor 欠薪：命中第91、50条，排名第1、第2；
- labor 违法辞退：命中第98条，排名第2；
- labor 社保：命中第72条，排名第2；
- civil 逾期交货：Top-5 miss；
- civil 借款到期不还：Top-5 miss；
- civil 邻居漏水：Top-5 miss。

因此当前主要问题不是“score 仍然全部为 0”，而是**embedding-only 对 civil 法条定位存在系统性失败**。大量相邻/相关主题法条挤占 Top-K。

## 当前对结果的正确解释

### 1. Hybrid 当前没有证据证明有效

此前 hybrid A/B 没有带来额外命中，因此不能宣称 n-gram lexical overlap 可以解决当前问题。它仍可作为一个可选实验，但不是当前优先项。

### 2. Civil 是当前 retrieval 的短板

买卖合同逾期交货、借款到期不还、邻里漏水三个 civil query 都没有在 Top-5 找到 gold。要进一步区分：

- gold 是否在 Top-20 候选内；
- 还是 embedding 从候选层面就没有召回 gold。

这决定后续是“rerank/ranking 问题”还是“embedding/query/index 问题”。

### 3. Reranker 必须做真实 A/B

`knowledge/retriever.py` 的 CrossEncoder rerank 现在会真正写回 `rerank_score` 并按该分数排序；但只有配置 `AGENT_DISTILL_RERANKER_MODEL` 才会执行。代码存在不代表模型可用或效果有效。

## 本阶段新发现：embedding 建库与查询必须完全一致

当前 `knowledge/ingest.py` 已使用：

```python
HuggingFaceEmbeddings(
    model_name=settings.embedding_model_name,
    encode_kwargs={"normalize_embeddings": True},
)
```

并创建：

```python
collection_metadata={"hnsw:space": "cosine"}
```

查询侧使用相同 normalized embedding contract。旧 collection 必须删除并重新建立，否则新代码不会改变旧向量。

## 必须继续做

### P0：重新建库

```powershell
uv run python -m knowledge.ingest --reset
```

然后检查 collection metadata 和少量 query 的 distance/score。

### P0：Top-5/10/20 曲线

对同一 gold benchmark 运行：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --output data/evaluation/retrieval_top5.json
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 10 --output data/evaluation/retrieval_top10.json
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 20 --output data/evaluation/retrieval_top20.json
```

目标不是靠增大 K 掩盖低 Precision，而是判断 civil gold 到底有没有进入更大候选池。

### P0：Reranker A/B

配置可用的本地 CrossEncoder 后，对同一 gold set 跑：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rerank --output data/evaluation/retrieval_rerank_top5.json
```

并与 embedding-only 比较 Precision@5、Recall@5、MRR，尤其查看 civil 三个 case。

### P1：确认 score threshold

当前 Runtime 的 `retrieval_min_score=0.45` 只能作为工程初始值。必须在新的 cosine collection 上取得真实 score 分布后重新校准；不能继续沿用旧 L2 score 的量级判断。

### P1：Civil query expansion

只有在 Top-20 确认 gold 存在、但排名靠后时，才优先实验 query expansion，例如：

- `逾期交货` → `合同迟延履行 / 违约责任`；
- `借款不还` → `借款期限 / 到期返还 / 利息`；
- `邻居漏水` → `相邻关系 / 侵权责任 / 财产损失赔偿`。

必须在整个 benchmark 上评估，不能针对 3 个样本硬编码。

### P1：Civil metadata/topic labels

如果 query expansion 仍不稳定，可以给 article 增加 topic/scenario metadata，再做 metadata-assisted retrieval A/B。

### P1：Embedding 模型 A/B

只有在 Top-20 仍 miss gold，才把 embedding 模型作为主要变量。保持同一 chunk、同一 gold、同一 cosine metric，比较候选模型。

### P1：Citation semantic grounding

当前只验证 citation 来源和法条编号匹配，不验证法律语义解释正确性。后续增加 entailment 或人工一致性评估。

### P1：失败 trajectory quarantine

失败/空 answer 可以保留到 rejection 数据用于 error analysis，但正式 Decision/Answer SFT 不得使用这些样本。

### P2：PDF source integrity

跨页 article-boundary 已修复，但仍需抽样检查原 PDF 是否有 OCR/字体/页眉页脚污染、条款缺字和超长条款二次切分问题。

## 当前推荐实验顺序

```text
pytest -q
→ ingest --reset
→ 检查 hnsw:space / distance / score
→ Top-5 / Top-10 / Top-20
→ Reranker A/B
→ 判断 civil 是 recall 问题还是 ranking 问题
→ 必要时 query expansion
→ 必要时 embedding A/B
→ 重新生成 20~50 条 trajectory v2
→ 人工抽样
→ prepare_phase5_data.py
→ 最后训练 Decision/Answer LoRA
```

## 明确禁止

- 不要用旧 `agent_trajectory.jsonl` 直接训练最终 LoRA。
- 不要把 `score` 当 calibrated probability。
- 不要在没有 benchmark 的情况下把 threshold 从 0.45 改到 0.7。
- 不要让 LLM 自动补全文本并写回知识库。
- 不要只看 Recall@K 忽略 Precision@K。
- 不要在没有 A/B 证据时宣称 hybrid 或 reranker 有效。

## 当前分支

工作分支：`fix/phase6`。`master` 不应被本阶段修改污染；LoRA 大文件不提交。

最新关键代码：

- `knowledge/ingest.py`：normalized embedding + cosine collection；
- `knowledge/retriever.py`：distance/score/rerank_score + hybrid；
- `mcp_service/retriever_service.py`：统一读取 rerank / threshold 配置；
- `evaluation/retrieval_benchmark.py`：gold retrieval benchmark；
- `evaluation/router_benchmark.py`：open-set router benchmark。

本报告必须随阶段进展持续更新，尤其要记录新的 benchmark 数字、真正完成的实验和未完成事项。
