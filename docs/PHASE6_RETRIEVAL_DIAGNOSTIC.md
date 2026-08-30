# Phase 6 Retrieval 诊断与交接报告

> 分支：`fix/phase6`。下一个 AI/开发者应先阅读本文，不要直接重新训练 LoRA。

## 当前结论
随机抽样 trajectory 已暴露：领域路由错误、检索噪声、旧 score 全为 0、citation/verification 边界不足、空 answer，以及 PDF 跨页法条截断。旧 trajectory 只能用于 error analysis，不应直接作为最终 SFT。

## 已完成

- Router：增加 `unknown`/abstention、条件路由以及 labor/civil 规则；测试通过。
- Citation：`retrieved_documents` 与 `citations` 分离；答案引用不能超出检索证据。
- Retrieval：candidate pool + 可选 CrossEncoder；保留 `distance`、`score`、`rerank_score`。
- Score：不再使用会把 L2 distance 大于 2 全压成 0 的 `1-distance/2`。当前 L2 诊断分数为 `1/(1+distance)`，cosine 为 `1-distance`。这些不是概率，也没有校准。
- PDF chunking：增加 whole-PDF/article-boundary 切分；跨页条款不再因为 page boundary 被直接截断。
- Phase 5：Decision/Answer SFT 分离；过滤空 answer 和 `verification.passed == false` 的 trajectory。
- 新增 `evaluation/retrieval_benchmark.py` 和 `data/evaluation/retrieval_benchmark_v2.jsonl`，用 canonical `law_name + article` 做 gold retrieval 评估。

## 重要注意

`data/evaluation/benchmark.jsonl` 仍有历史 retrieval gold 占位符，例如 `civil_loan_regulation`、`labor_wage_regulation`，不能直接与新的 canonical reference 比较。

## 必须继续做

### P0：重新构建 Chroma

由于 ingest 切分策略已经改变，旧 collection 可能含有旧 chunks。应执行：

```powershell
uv run python -m knowledge.ingest --reset
```

否则新旧 chunks 可能混在一起。

### P0：运行 Retrieval benchmark

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --output data/evaluation/retrieval_v2_report.json
```

分别记录 Precision@K、Recall@K、MRR、distance、score、rerank_score。先不要凭经验修改阈值 `0.45`。

### P0：Reranker A/B

只有配置 `AGENT_DISTILL_RERANKER_MODEL` 后才会真的使用 CrossEncoder。必须在同一 gold 集上比较 embedding-only 与 reranker 的 Precision/Recall/MRR。

### P0：Router gold benchmark

需要独立覆盖 labor、civil、non-legal、ambiguous，尤其包括“邻居漏水→civil”“天气/寒暄→unknown”等 hard cases。

### P1：Citation 语义一致性

当前只保证 citation 来自 retrieval 且答案显式提到法条编号，尚不能保证法条含义被正确解释。后续增加 entailment/人工审核。

### P1：失败 trajectory

原始 trajectory 可以保留失败记录做分析，但最终训练集必须继续过滤空 answer / verification failure；不能为了数据量把失败样本重新混入 SFT。

### P2：PDF 完整性

跨页切分已修复，但仍需抽样检查原 PDF 文本是否本身缺字、错字、页眉页脚污染，并增加 article completeness tests。

## 推荐执行顺序

```text
pytest -q
→ knowledge.ingest --reset
→ 检查 Chroma hnsw:space
→ retrieval_benchmark_v2
→ reranker A/B
→ router gold benchmark
→ 重新生成 20~50 条 trajectory
→ 人工复核
→ prepare_phase5_data.py
→ 最后训练 Decision/Answer LoRA
→ 新 Raw vs LoRA benchmark
```

## 明确未完成

1. Retrieval threshold 未完成 benchmark 校准。
2. Reranker 未完成真实 A/B。
3. Router 未完成独立 gold benchmark。
4. Citation 未完成法律语义 entailment。
5. PDF 源文本完整性未全面验证。
6. 正式 trajectory v2 未重新生成并人工审核。
7. 新 Decision/Answer LoRA 未训练。
8. 新 Raw vs LoRA 对比未开始。

## 当前最重要的交接结论

**先证明 Retrieval 是好的，再训练 LoRA。** 如果 gold benchmark 显示 Recall/Precision 仍低，优先修 Retrieval；只有检索质量稳定后，LoRA 的效果比较才有意义。
