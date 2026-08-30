# Phase 6 Retrieval 诊断与交接报告

> 分支：`fix/phase6`
>
> 本文是持续交接文档。下一个 AI/开发者应先阅读本文，再继续修改；**在 Retrieval 验收完成前不要重新训练 LoRA**。

## 1. 当前结论

Phase 6 的核心瓶颈已经从单纯的 LoRA 问题收敛为 supervision/data pipeline 问题：Router、Retrieval、Citation、Verification 任一环节不可靠都会污染 trajectory。当前最重要的是先把知识库检索质量建立成可量化、可复现的 baseline。

历史随机轨迹曾发现：

- 邻里漏水等典型民事侵权问题被错误路由到 labor；
- Retrieval 人工相关率约 20%~40%；
- `score` 曾全部为 `0.0`；
- `citations` 曾复制所有 `retrieved_documents`；
- answer 与 citation 可能不一致；
- verification 过松；
- generation 失败可能生成空 answer trajectory；
- PDF 存在跨页/相邻法条截断和合并问题。

因此：**旧 trajectory 不可直接作为最终 SFT 数据。**

## 2. 已完成修复

### 2.1 Router

已加入 `unknown` / abstention、条件路由以及 labor/civil hard rules。非法律问题不再强制进入法律检索；邻居漏水、交通事故、借款、租赁等典型 civil 场景增加了规则覆盖。现有测试已经通过，但尚未完成独立 Router gold benchmark。

### 2.2 Citation / Verification

已区分：

- `retrieved_documents`：候选证据；
- `citations`：回答实际引用的证据。

验证会检查 answer 非空、citation 来源、答案中的法条编号等；同时兼容旧格式中法条编号只存在于 reference 的情况。当前仍不足以证明法律语义上的 entailment。

### 2.3 法条切块

已从逐页独立切分升级为 whole-PDF article-boundary 切分。跨页法条不会再因为 page boundary 被直接截断；相邻法条也会按 `第X条` 拆分。

### 2.4 Chroma score

历史实现使用 `1-distance/2` 并 clamp，这会把较大的 L2 distance 大面积压成 0。现在保留原始 `distance`，根据 collection metric 计算诊断 `score`，并保存可选 `rerank_score`。当前 L2 使用 `1/(1+distance)`，cosine 使用 `1-distance`。这些均为排序/诊断分数，不是概率。

### 2.5 Embedding / Collection

当前 ingest 已统一使用：

- `normalize_embeddings=True`；
- Chroma `hnsw:space = cosine`。

因此需要在本地重新 `--reset` 构建 collection，否则旧 L2 collection 与新配置不一致。

### 2.6 Retrieval hybrid

Retriever 增加了可选 hybrid 模式：

```text
semantic candidate retrieval
        +
Chinese n-gram lexical overlap
        ↓
combined ranking
```

若配置 `AGENT_DISTILL_RERANKER_MODEL`，还可以继续使用 CrossEncoder 二阶段排序；其分数写入 `rerank_score`。

MCP Runtime 已开启 `hybrid=True`，并继续做相关性门控。

### 2.7 训练数据隔离

`distill/prepare_phase5_data.py` 已过滤空 answer 和 `verification.passed == false`；失败 trajectory 另有 `distill/filter_trajectory.py` 做隔离测试。目标是：失败样本可以保留用于 error analysis，但不能进入最终 SFT。

## 3. 真实 Retrieval Benchmark 结果

用户本地在新 benchmark 上运行 embedding-only retrieval，得到：

```text
samples = 7
top_k = 5
rerank = false
precision_at_k = 0.142857
recall_at_k = 0.571429
mrr = 0.333333
```

逐条现象：

- labor 加班：gold 第44条命中，但排名第3；前面还有第50/51/100/45条。
- labor 欠薪：gold 第91、50条均命中，排名分别第1、第2；这一条表现最好。
- labor 违法辞退：gold 第98条命中，排名第2。
- labor 社保：gold 第72条命中，排名第2。
- civil 逾期交货：**Top-5 0 命中**。
- civil 借款到期不还：**Top-5 0 命中**。
- civil 邻居漏水：**Top-5 0 命中**。

因此当前问题不能只描述成“分数不对”：**民法典检索在当前 embedding-only baseline 上出现系统性失败。** 现在 Recall@5=0.571 且 Precision@5=0.143，说明候选中有部分命中，但大量 top-k 被相邻法条占据。

本次报告中的上述数字来自真实本地 benchmark，不能直接在仓库环境复现，因为 Chroma 数据库和本地 embedding 权重不随代码库提交。

## 4. 当前最重要的新发现

### 4.1 旧 score=0 已不是主要问题

本次 benchmark 的 `distance` 大约位于 188~276，经过新的 L2 诊断变换后得到约 0.0036~0.0053。这证明数值已经不再全部为 0，但**动态范围很窄**，所以这个 score 不能直接作为“相关性概率”或固定阈值依据。

### 4.2 当前 embedding-only 排序有明显 domain-specific failure

labor query 可以召回部分 gold article，但 civil 三个 query 全部 Top-5 miss。优先怀疑：

1. embedding 模型对法律条款语义的区分不足；
2. query 与法条原文的表达差异较大；
3. 民法典中的相邻条文主题连续，向量召回容易选择相邻条款；
4. 需要 hybrid/reranker，而不是继续拍 score threshold。

### 4.3 score threshold 目前不能继续凭经验调

因为当前 L2 score 只有约 0.004 的量级，历史 `0.45` threshold 对这种 score 空间没有实际校准依据。如果直接按 `score >= 0.45` 过滤，可能会把**全部候选都过滤掉**。

因此下一步要把“threshold”从原始 semantic score 中解耦，优先使用 rank/hybrid/reranker，再通过 gold benchmark 校准最终门槛。

## 5. 本次新增代码

### `knowledge/ingest.py`

- normalized embeddings；
- cosine collection metadata；
- whole-PDF article-aware ingestion。

### `knowledge/retriever.py`

- `distance` / `score` / `rerank_score`；
- normalized query embeddings；
- optional hybrid lexical-semantic ranking；
- candidate pool。

### `mcp_service/retriever_service.py`

- Runtime 默认开启 hybrid；
- score 保留；
- 相关性 gate；
- 输出 `score` / `rerank_score`。

### `evaluation/retrieval_benchmark.py`

新增 `--hybrid` 参数，允许：

```text
embedding-only
vs
hybrid
vs
hybrid + CrossEncoder
```

做可重复 A/B。

### `tests/test_retriever_hybrid.py`

覆盖中文短语 overlap 和 hybrid 排序。

## 6. 尚未完成

### P0：重新 ingest 新 collection

执行：

```powershell
uv run python -m knowledge.ingest --reset
```

这是必须步骤。否则当前本地 Chroma 可能仍是旧 L2 / 未归一化 embedding。

### P0：重新跑三组 Retrieval baseline

1. embedding-only：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --output data/evaluation/retrieval_embedding_v2.json
```

2. hybrid：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --hybrid --output data/evaluation/retrieval_hybrid_v2.json
```

3. hybrid + CrossEncoder（本地配置 `AGENT_DISTILL_RERANKER_MODEL` 后）：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --hybrid --rerank --output data/evaluation/retrieval_rerank_v2.json
```

### P0：解决当前 relevance gate 与新 score 空间的不一致

不能继续使用固定 `0.45` 直接解释新的 cosine/L2 score。应先比较三组 benchmark，然后基于真实 score distribution 或 reranker score 选择 gate。

### P0：建立 Router gold benchmark

至少：

- labor 20+
- civil 20+
- non-legal 20+
- ambiguous 10+

统计 Accuracy、Macro-F1、unknown precision/recall、confusion matrix。

### P1：改进 query formulation

如果 hybrid 和 reranker 仍不能把 civil gold article 拉进 Top-5，需要增加 legal query normalization / expansion，例如：

- `逾期交货` ↔ `迟延履行 / 违约责任`
- `借款不还` ↔ `借款到期 / 返还借款 / 利息`
- `邻居漏水` ↔ `相邻关系 / 侵权责任 / 财产损害赔偿`

扩展规则必须通过 benchmark 验证，不能只针对 3 个样本硬编码。

### P1：确认 Reranker 是否可用

目前 CrossEncoder 只是可选依赖；未配置时不会真正执行。必须完成 A/B 证据，不要假设“代码支持”就等于“效果已改善”。

### P1：Citation semantic grounding

当前验证只能确认 citation 来源及法条编号，尚不能证明答案对法条含义解释正确。后续需要 entailment 检测或人工标注。

### P1：失败 trajectory quarantine

保留失败轨迹用于 error analysis，但正式 SFT 只允许通过质量门槛的记录。

### P2：PDF source integrity

跨页切分已经修复，但仍需检查原 PDF 是否存在 OCR/字体/页眉页脚造成的缺字和污染，并增加 article completeness tests。

## 7. 训练 LoRA 前的硬门槛

在满足以下条件前，不开始新的 Decision/Answer LoRA 训练：

```text
[ ] Router gold benchmark 通过
[ ] Retrieval embedding baseline 可解释
[ ] Retrieval hybrid A/B 完成
[ ] CrossEncoder A/B 完成（若可用）
[ ] Precision@5 不再低到无法作为 evidence source
[ ] Recall@5 达到可接受水平
[ ] Citation / verification 能拦截明显 mismatch
[ ] 空 answer / failed trajectory 已隔离
[ ] 新 trajectory 20~50 条人工抽样通过
```

## 8. 交接结论

**当前真正的问题是 Retrieval quality，而不是继续调 LoRA。** 新 benchmark 已证明旧 score=0 是一个独立 bug，但修完 score 后仍只有 Precision@5≈0.143、Recall@5≈0.571，且 civil 三个样本全部 miss。下一阶段应围绕 `cosine normalized embedding → hybrid lexical → CrossEncoder → query expansion` 做实验，并用 gold benchmark 决定最终方案。

## 9. 当前分支状态

工作分支：`fix/phase6`

保留 `master` 不变；不提交 LoRA 大文件。

本阶段新增 retrieval/hybrid 代码和本交接报告，数据和旧实验结果继续保留在仓库中。
