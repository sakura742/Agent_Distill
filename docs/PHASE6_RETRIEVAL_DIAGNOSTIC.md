# Phase 6 Retrieval 诊断与交接报告

> 分支：`fix/phase6`
>
> 本文用于交接。下一个 AI/开发者应先阅读本文，再继续修改，不要直接重新训练 LoRA。

## 1. 当前结论

Phase 6 当前最大风险仍然是数据生成链路，而不是 LoRA 超参数。随机抽样 trajectory 已发现：

- domain routing 曾把邻里漏水等典型民事侵权问题错误路由到 labor；
- embedding retrieval 的人工相关率约 40%，存在明显噪声；
- trajectory 中曾出现 `score: 0.0` 全部相同，导致无法判断检索质量；
- `citations` 与 `retrieved_documents` 已经分离，但历史数据不能直接视为高质量 supervision；
- verification 已增强为检查 answer/citation 基本一致性，但仍不是法律语义 entailment；
- generation 失败时仍可能产生空 answer trajectory，这类样本不应进入最终 SFT；
- 部分 PDF 条款存在截断或相邻条款污染，仍需源文档级修复。

因此：**在 Router、Retrieval、Citation、Verification 和 trajectory 清洗验收前，不重新训练 LoRA。**

## 2. 已完成

### 2.1 Router

已加入 non-legal/unknown abstention，并增加 labor/civil 典型领域规则，降低明显领域问题被 embedding 错分的概率。当前测试已经通过。后续仍需用独立 gold benchmark 测量，而不能只依赖单元测试。

### 2.2 Citation / Verification

已区分：

- `retrieved_documents`：检索得到的候选证据；
- `citations`：回答实际声明引用的证据。

验证会检查 answer 非空、citation 来源以及答案中的法条编号等。兼容旧测试中法条编号只存在于 reference 的情况。

### 2.3 Retrieval candidate / rerank

Retriever 支持 candidate pool 和可选 CrossEncoder rerank；配置通过：

- `AGENT_DISTILL_RERANKER_MODEL`
- `AGENT_DISTILL_RETRIEVAL_RERANK`
- `AGENT_DISTILL_RETRIEVAL_CANDIDATE_MULTIPLIER`
- `AGENT_DISTILL_RETRIEVAL_MIN_SCORE`

控制。

## 3. 本次新增：Chroma score 修复

此前 `knowledge/retriever.py` 使用：

```python
1.0 - distance / 2.0
```

再 clamp 到 `[0, 1]`。这不是通用的 Chroma distance → similarity 转换：L2 distance 大于 2 时会全部被压成 0，因此无法解释或比较检索质量。

本次改为：

- 保留原始 `distance`；
- 根据 collection 的 `hnsw:space` 判断 metric；
- cosine：`score = 1 - distance`；
- L2：使用有界单调变换 `score = 1 / (1 + distance)`；
- reranker 启用时另外保存 `rerank_score`。

注意：这些 score 是**排序/诊断分数，不是概率，也不是已经校准的相关性概率**。因此 `0.45` 阈值仍然只是工程初始值，必须用 gold benchmark 校准。

## 4. 当前必须继续做的工作

### P0：真实 Retrieval Benchmark

建立独立 gold 数据集，至少覆盖 labor/civil/non-legal/ambiguous，并为 legal query 标注 `gold_references`。

统计：

- Precision@1/3/5
- Recall@1/3/5
- MRR
- 平均检索 score
- score 分布
- 不同 domain 的混淆

不要用生成后的 answer 反向制造 gold，否则会产生评估泄漏。

### P0：验证 Router

重点加入：

- 邻居漏水 → civil
- 交通事故 → civil
- 借款/租赁/侵权 → civil
- 加班/工资/劳动合同/裁员 → labor
- 天气/寒暄 → unknown
- 模糊边界 → unknown 或低置信度

同时检查 rule 与 embedding 冲突时的行为。

### P0：空 answer 样本隔离

trajectory generation 流程应改为：

```text
generation
  -> verification/retry
  -> still empty or failed
  -> discard from training dataset
```

而不是把失败样本继续写进最终 SFT 文件。建议保留单独的 failure log，供 error analysis 使用。

### P1：Retrieval metric 与 threshold 校准

先获取真实 Chroma distance/score 分布，再决定 `retrieval_min_score`。不要凭经验把 0.45 改成 0.7。

同时确认实际 collection metric。重新 ingest 后需要确认 collection metadata 是否为预期的 L2/cosine。

### P1：Reranker 实测

目前代码支持 CrossEncoder，但只有配置本地 reranker 后才真正执行。需要对同一批 gold query 做 A/B：

```text
embedding TopK
vs
embedding candidate pool + reranker TopK
```

只有 Precision/Recall/MRR 实际改善，才保留 reranker 作为默认路径。

### P1：Citation grounding

当前已经检查 citation 是否来自 retrieval、答案是否出现法条编号，但仍不能保证“答案对法条的解释正确”。后续可加入轻量 evidence entailment 检查或人工标注。

### P2：PDF 条款完整性

发现过第91条等内容截断，以及多个相邻法条被同一 chunk 包含的问题。`knowledge/legal_chunker.py` 已按 `第X条` 做条款级切分，但如果原 PDF 页面文本本身缺失，chunker 无法凭空恢复全文。

下一步应：

1. 对原始 PDF 做页面级抽样；
2. 与官方/可信法律文本比对；
3. 修 parser；
4. 增加 article completeness tests；
5. 必要时重新 ingest Chroma。

**禁止让 LLM 自动补全文本并写回法律知识库。**

## 5. 重新生成 trajectory 前的验收门槛

必须至少满足：

```text
Router benchmark 通过
Retrieval benchmark 有真实 score
Precision@K / Recall@K 可解释
Citation 是 retrieved subset
Answer 引用不能超出 retrieved evidence
Verification 能拦截明显 evidence mismatch
失败 generation 不进入训练集
20~50 条新 trajectory 人工抽样通过
```

然后才能生成正式 `agent_trajectory_v2.jsonl`。

## 6. 推荐执行顺序

```text
1. 拉取 fix/phase6
2. pytest -q
3. 检查 Chroma collection metric
4. 用真实 query 打印 distance/score
5. 建 retrieval gold benchmark
6. A/B embedding vs reranker
7. 完成 empty-answer discard
8. 完成 citation/evidence verification
9. 重新生成 20~50 条 trajectory
10. 人工复核
11. 生成 Decision/Answer SFT
12. 最后才训练 Qwen3.5-4B LoRA
```

## 7. 重要历史问题

旧 `agent_trajectory.jsonl` 中已经出现过以下污染模式：

```text
错误 domain
错误 collection
低相关法条
citations 全量复制 retrieval
answer 引用未检索法条
verification 仍通过
answer 为空
```

因此旧 trajectory 应保留用于 error analysis，但**不能直接当作最终高质量训练集**。

## 8. 本次提交说明

本阶段新增/修改：

- `knowledge/retriever.py`：修正 distance/score 表达并保留 `distance`、`rerank_score`；
- `docs/PHASE6_RETRIEVAL_DIAGNOSTIC.md`：本交接报告。

最新提交：`e28e09916cdf9e7eecac8706a7b209b171440895`。

后续开发应以这个提交为起点继续，不要假设 Retrieval benchmark、reranker A/B、empty-answer discard、法律语义验证已经完成。
