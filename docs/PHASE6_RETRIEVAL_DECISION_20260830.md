# Phase 6 Retrieval 决策记录（2026-08-30）

## 当前实验结论

用户最新 A/B 结果显示：

- 纯 GTE-large-zh：Precision@5=0.171，Recall@5=0.786，MRR=0.452；
- GTE + 旧版 Rewrite：Precision@5=0.143，Recall@5=0.643，MRR=0.381；
- 改为 additive rewrite（原始 query 与 rewrite query 各自召回后取并集）后，#005/#006 均重新获得有效命中；具体最终均值必须以重新跑的 benchmark JSON 为准，不能仅根据单条样本推断。
- Reranker 历史实测为负：MRR 0.405 -> 0.179，Recall 0.714 -> 0.429，因此当前不作为默认链路。
- #007「楼上邻居漏水」仍是核心 hard case。

## 关键架构判断

### Rewrite

Rewrite 不应替换原始 query，只能增加候选。原因是 #005 原始 query 已经能命中第577条，而替换式 rewrite 会把该命中挤出候选池。

### Reranker

Reranker 只能作为显式实验开关。必须保证：

1. 本地 reranker 模型真实加载；
2. `rerank_score` 非空；
3. 排序真正使用 `rerank_score`；
4. 实验记录 embedding-only 与 rerank 两组结果。

### #007

当前不能仅凭“Reranker 能否把 #007 提升到 Top-5”判断问题。首先必须回答：Gold 第1165/1184条是否进入 candidate pool。

- 如果 Top-50 candidate pool 仍没有 gold：这是召回/索引/查询表示问题，Reranker 无法解决；
- 如果 Top-50 有 gold 但 Top-5 没有：才是 ranking 问题，Reranker 才有救援空间。

## 本轮新增诊断工具

`evaluation/retrieval_diagnostics.py`

用途：只运行 embedding candidate retrieval，不做 rerank，用 `candidate_k=50` 或更大值判断 gold 是否可达。

示例：

```powershell
uv run python -m evaluation.retrieval_diagnostics data/evaluation/retrieval_benchmark_v2.jsonl --candidate-k 50 --embedding-model "thenlper/gte-large-zh" --chroma-dir "knowledge/chroma_gte_large_zh" --output data/evaluation/candidate_pool_v2.json
```

重点查看 #005/#006/#007 的：

- `gold_in_candidate_pool`
- `gold_ranks`

## 当前默认策略

正式 Agent 不启用 Reranker。

Rewrite 是否默认启用不应凭最近一次旧实验决定；当前最可靠 baseline 仍是纯 GTE，additive rewrite 需要重新跑完整 benchmark 后才能决定是否升级为默认。

建议配置：

```text
Embedding = GTE-large-zh
Reranker = OFF
Hybrid = OFF（除非单独 A/B 证明有效）
Rewrite = A/B only，默认保持 OFF
```

## 下一阶段

1. 跑 `retrieval_diagnostics.py --candidate-k 50`；
2. 确认 #007 gold 是否可达；
3. 若不可达，进入 Civil index/topic enrichment；
4. 若可达，继续 reranker ranking 实验；
5. 重新生成 trajectory v2 前，必须满足 retrieval 和 verification 验收；
6. 空答案/错误 citation 样本不能进入正式 SFT；
7. Retrieval 未稳定前不要重新训练 LoRA。

## 禁止事项

- 不要把“Reranker 已安装/代码支持”写成“Reranker 已有效”；
- 不要把 score 当概率；
- 不要因为 Top-K 增大就宣称 recall 已解决；
- 不要为 #007 添加单独写死的法条规则以抬高 benchmark；
- 不要把 benchmark gold 泄漏到索引文本中。
