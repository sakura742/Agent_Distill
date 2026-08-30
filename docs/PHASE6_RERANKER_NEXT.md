# Phase 6：GTE Baseline → Reranker 阶段交接

## 1. 当前结论

经过同一 retrieval benchmark 的 embedding A/B：

- `thenlper/gte-large-zh` 在当前法律语料上的表现优于旧 `shibing624/text2vec-base-chinese`，正式作为 Retrieval baseline。
- GTE 成功解决 `#005 买卖合同逾期交货 -> 民法典第577条` 的候选召回问题，说明旧 embedding 在该类语义边界上确有不足。
- `#007 邻居漏水 -> 民法典第1165/1184条` 在已有 embedding + rewrite/rerank 前置实验中仍是 hard case；不能因为领域接近就假设已经解决。
- Query rewrite 对借款类问题有明显收益，应保留为一个独立可控变量。

## 2. 为什么当前不继续盲目换 embedding

#005 已经被 GTE 拉回候选集，说明继续无目标地更换 embedding 的边际收益需要证据支持。
当前首要任务转为二阶段排序：对 embedding top-20 candidate 使用 CrossEncoder，将 query 与法条 passage 进行 pairwise relevance scoring。

## 3. Reranker 实验目标

重点验证三个问题：

1. `#006 借款` 是否能从 Top-10/Top-20 提升到 Top-5；
2. `#007 漏水` 的 gold 是否已经进入 candidate pool，如果进入，reranker 能否提升排名；
3. reranker 是否损害已经表现良好的 labor case 和 GTE 的 #005。

## 4. 推荐模型

优先实验本地可加载的 `BAAI/bge-reranker-v2-m3`。项目不内置模型权重，只通过配置指定名称或本地目录。

## 5. 实验控制变量

每组必须保持一致：

```text
same PDF corpus
same article-level chunking
same gold benchmark
same embedding = GTE-large-zh
same candidate_k = 20
same top_k = 5
same query rewrite state
```

只切换：

```text
rerank = false
vs
rerank = true
```

必要时单独做：

```text
rewrite = false
vs
rewrite = true
```

不要同时修改 embedding、chunking、rewrite、reranker 后再声称某一个组件带来提升。

## 6. Reranker 必须真正被证明启用

代码层面 `rerank_score` 存在并不等于 reranker 已执行。运行前必须配置：

```powershell
$env:AGENT_DISTILL_RERANKER_MODEL="D:\py\embeddings\bge-reranker-v2-m3"
```

或使用可离线解析的模型路径/名称。

然后运行：

```powershell
uv run python -m evaluation.reranker_diagnostics data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --candidate-k 20 --output data/evaluation/reranker_diagnostic.json
```

检查：

```text
reranker_model_configured = true
每条 sample 的 reranker_scores 至少一个非 null
```

若全部为 null，不得将该结果标记为 reranker 实验。

## 7. Retrieval A/B

Baseline：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh --output data/evaluation/gte_top5.json
```

Rerank：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rerank --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh --output data/evaluation/gte_rerank_top5.json
```

Rerank + rewrite：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --rerank --rewrite --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh --output data/evaluation/gte_rerank_rewrite_top5.json
```

## 8. 结果判定

### A. `#006` 从 Top-10/20 提升进 Top-5

说明主要是 ranking 问题，reranker 有价值。

### B. `#007` gold 在 candidate_top20，但 Top-5 仍无

说明 candidate recall 足够，但细粒度 semantic ranking 仍不足；下一步考虑法律场景 metadata/topic labels 或更强的 legal reranker。

### C. `#007` 连 Top-20 都没有

说明 reranker 无法解决 candidate recall 问题。下一步转向：

- Civil query formulation；
- article topic/scenario metadata；
- legal-domain embedding / retrieval model A/B；
- 必要时人工构建 hard-negative benchmark。

## 9. 训练门槛

仍然禁止重新训练 LoRA，直到：

```text
[ ] GTE baseline 已在 clean rebuilt index 上复现
[ ] Reranker 已证明实际启用
[ ] #006 regression 已检查
[ ] #007 candidate recall 已确认
[ ] Retrieval Precision/Recall/MRR 达到项目预设门槛
[ ] trajectory v2 人工抽样合格
```

## 10. 明确当前未完成

- Reranker 真实本机 A/B 尚未完成；
- `#007` 尚未解决；
- Civil topic/scenario metadata 尚未实现；
- citation semantic entailment 尚未实现；
- 新 trajectory v2 尚未作为正式 SFT 数据集生成；
- Decision/Answer LoRA 尚未重新训练。

## 11. 当前分支

工作分支：`fix/phase6`。

不提交 LoRA/model weights；只提交代码、benchmark、数据和诊断报告。
