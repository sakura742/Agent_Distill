# Phase 5 trajectory 数据质量问题报告

## 结论

随机抽样轨迹显示，当前最大风险不是 LoRA 超参数，而是 supervision pipeline：领域路由、检索精度、引用选择和验证逻辑会把噪声写入 `agent_trajectory.jsonl`。在这些问题解决并重新生成数据前，不应把旧 trajectory 直接作为高质量 SFT 数据。

## 已修复

### P0：非法律问题强制进入法律检索

Router 增加 `unknown` / abstention，Runtime 增加条件路由。非法律问题不再调用法律工具。

### P0：`retrieved_documents` 与 `citations` 混淆

retrieval 节点只产生 `retrieved_documents`，generation 初始化 `citations=[]`，再根据答案中明确出现的法条编号，从已检索证据中选择实际使用的 citation。

### P0：相邻法条合并

Runtime parser 会按“第X条”标题拆分包含连续法条的 chunk，使每条法条可以独立引用。

### P1：verification 过简

法律回答现在至少要求：answer 非空、存在 citation、citation 必须来自 retrieved documents、答案中必须出现被引用法条编号。非法律回答要求无 tool、无 citation 且 answer 非空。

### P1：检索结果增加相关性控制

Retriever 本身已经提供 `RetrievedChunk.score`、candidate pool 和可选 CrossEncoder rerank。MCP service 现在使用候选集、启用 rerank，并对带 score 的结果应用初始最低相关性门槛 `0.45`，同时将 score 写入 tool result，供 trajectory 保留。

> 注意：`0.45` 是当前工程初始值，不是最终标准。必须通过 retrieval benchmark 校准。

## 尚未完全解决

### P1：Retriever score / threshold 需要 benchmark 校准

Chroma 的 distance 被当前代码映射到 `[0,1]` score，但不能假设 `0.45` 或 `0.70` 在所有 embedding / metric 下都是正确阈值。下一步应建立有 gold reference 的 retrieval benchmark，统计 Precision@1/3/5、Recall@1/3/5、MRR，并据此选择阈值。

### P1：Reranker 模型是否真正启用

系统支持 CrossEncoder，但只有配置 `AGENT_DISTILL_RERANKER_MODEL` 时才会实际加载模型。未配置时保持 embedding 排序。因此重新生成 trajectory 前应确认环境变量已经指向本地可用 reranker，避免误以为已经完成二阶段排序。

### P2：PDF 法条截断

第91条只出现部分条文等问题属于源 PDF 解析/切块问题。不能让 LLM 自动补全文本，否则会制造训练集幻觉。需要检查原始 PDF、parser 和 chunking，并增加法条完整性测试。

### P1：Citation 的法律语义正确性

当前验证可以确认 citation 来自 retrieval，并且答案明确提到对应法条编号，但还不能证明答案对法条含义的解释完全正确。后续可以增加 evidence entailment / citation consistency 检查。

## 重新生成 trajectory

旧 `agent_trajectory.jsonl` 应作为历史实验数据保留，不直接覆盖使用。修复后的 pipeline 应重新生成，例如：

```powershell
uv run python -m distill.trajectory distill/data/phase5_raw_questions.jsonl --output distill/data/agent_trajectory.jsonl
```

默认覆盖；只有显式 `--append` 才追加。

建议先抽样 20~50 条人工复核，再生成最终 Decision / Answer SFT 数据。

## 验收标准

1. labor/civil/non-legal/ambiguous Router 均有独立测试。
2. Retrieval 报告 Precision@1/3/5、Recall@1/3/5、MRR。
3. `citations` 是 `retrieved_documents` 的子集。
4. non-legal：`tool_name` 为空、`retrieved_documents=[]`、`citations=[]`。
5. legal：每个 citation 能在 answer 中定位到明确法条编号。
6. 相邻法条必须拆成独立 reference。
7. 检索 score 和 reranker 状态在 trajectory 中可观察。
8. 抽样人工复核通过后，才重新训练 Decision LoRA / Answer LoRA。
