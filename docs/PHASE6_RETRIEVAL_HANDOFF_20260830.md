# Phase 6 当前阶段交接补充（2026-08-30）

## 已知 benchmark 结果
用户本地 Retrieval embedding-only Top-5：Precision@5=0.142857，Recall@5=0.571429，MRR=0.333333。Civil 的买卖合同、借款、邻居漏水曾全部 Top-5 miss；后续分析显示部分样本在更大的 K 才出现 gold，而部分样本 Top-20 仍 miss。

## 本轮确定的问题
1. Router 缺少借款/借钱/借条/民间借贷/债务人/欠款等 civil 触发词，已补充。
2. Query 与历史建库 embedding contract 曾不一致；ingest/query 现统一 normalized embeddings，并使用 cosine collection，必须重新 ingest --reset。
3. Chroma distance→score 原映射不可靠，现保留 distance 并使用 metric-aware diagnostic score；score 不是概率。
4. CrossEncoder rerank 已能真正按 rerank_score 排序，但没有配置本地模型时不会执行。
5. 已增加 deterministic Civil query normalization 模块，覆盖漏水、借款、逾期交货、押金、交通事故等法律术语扩展；该功能必须通过全量 benchmark A/B 后才可进入默认路径。
6. 旧 trajectory 仍不可信；失败/空 answer 不得进入正式 SFT。

## 本轮工程改动
- `agent/router.py`: civil 借款/借钱/借条等规则。
- `configs/settings.py`: 增加 rerank/hybrid/query-rewrite 实验开关和 rerank weight。
- `knowledge/query_rewrite.py`: 新增确定性法律 query normalization。
- `knowledge/retriever.py`: 已增加 query_rewrite 参数并保留 distance/score/rerank_score；注意后续版本需确认实际调用方是否已打开该开关。
- `tests/test_query_rewrite.py`: 新增 rewrite 回归测试。

## 下一步硬性顺序
1. 本地 `uv run python -m knowledge.ingest --reset`。
2. 检查 Chroma metadata `hnsw:space=cosine`，并打印真实 distance/score。
3. 跑 Top-5、Top-10、Top-20，并比较 civil gold 首次出现 rank。
4. 配置一个本地 CrossEncoder 后跑 embedding vs rerank A/B；没有可用模型时不要声称 rerank 已测试。
5. 跑 query_rewrite=false vs true A/B，检查 civil 改善和 labor regression。
6. 运行 Router gold benchmark，并记录 Accuracy/Macro-F1/unknown metrics。
7. 只有 retrieval + router 达标后才重新生成 trajectory v2。
8. 人工检查 20~50 条 trajectory；再生成 Decision/Answer SFT。
9. 最后训练 Qwen3.5-4B Decision LoRA 与 Answer LoRA。

## 不要做
- 不要直接修改 threshold=0.45/0.7 来制造指标提升。
- 不要用旧 `agent_trajectory.jsonl` 训练。
- 不要把 `score` 当相关性概率。
- 不要只针对三个 Civil 样本硬编码规则。
- 不要在没有真实 A/B 时声称 reranker 或 query rewrite 有效。

## 当前未完成
- Retrieval Top-5/10/20 新库结果尚未确认。
- Reranker 真实 A/B 未完成。
- Query rewrite A/B 未完成。
- Router gold benchmark 新结果未记录。
- Civil topic metadata 尚未建立。
- 更强 embedding 模型尚未 A/B。
- Citation semantic entailment 尚未完成。
- PDF source completeness 尚未全面验证。
- 新 trajectory v2 尚未正式生成。
- 新 Decision/Answer LoRA 尚未训练。

## 重要备注
`knowledge/retriever.py` 已增加 `query_rewrite` 参数，但当前 benchmark 是否默认开启取决于调用方；下一位开发者必须先确认 benchmark/runtime 的实际参数，避免实验标签与实际行为不一致。
