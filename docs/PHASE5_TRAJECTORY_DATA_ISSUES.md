# Phase 5 trajectory 数据质量问题报告

## 结论

`agent_trajectory.jsonl` 暴露的核心问题不是单一字段错误，而是检索→筛选→引用→验证链路缺少明确边界。当前已在 `fix/phase6` 修复数据生成与 Runtime 中最关键的两个问题：`retrieved_documents` 与 `citations` 分离，以及多法条 chunk 的拆分。与此同时，检索相关性阈值和原始 PDF 截断问题需要基于 Retriever / 文档解析器继续处理，不能仅靠 trajectory 层伪造修复。

## 已修复

### P0：citations 不能直接复制 retrieved_documents

此前 retrieval 节点把所有检索结果同时写入 `retrieved_documents` 和 `citations`。现在 retrieval 只产生 `retrieved_documents`，generation 根据回答中明确出现的法条编号/完整 reference 选择 `citations`。

如果回答没有实际引用某条检索证据，该证据不会进入 `citations`。

### P0：一个 chunk 中多个法条必须拆分

Runtime parser 现在会按 `第…条` 标题拆分相邻法条，并为拆出的记录生成独立 reference。这样类似第100—103条合并的问题不会继续污染 citation 粒度。

### P1：verification 不能只检查“有答案 + 有引用”

法律问题现在至少要求：答案非空、存在实际选中的 citation、citation 必须来自 retrieved_documents。非法律问题则要求没有 tool、没有 citation 且答案非空。

### P1：trajectory 输出安全性

`distill/trajectory.py` 现在默认覆盖输出文件，只有显式 `--append` 才追加；空输入和缺少 `input/question/user_query` 字段会直接报错，而不是静默生成 0 条 trajectory。

Teacher prompt 也明确要求只引用实际使用的法条，并在法律回答中写出法条编号。

## 尚未完全解决

### P1：检索相关性过滤 / rerank

当前 Runtime 能区分“检索到”和“实际引用”，但不能从现有 `tool_result` 中可靠获得原始 similarity score。因此不能在没有 Retriever score contract 的情况下硬编码 `score < 0.7`。

正确的下一步是让 Retriever 返回结构化结果：

```json
{
  "reference": "...",
  "content": "...",
  "score": 0.82
}
```

然后在 retrieval 层做 threshold + top-k/rerank。阈值必须通过 benchmark 校准，不能直接把 0.7 当作普适标准。

### P2：PDF 法条截断

第91条只出现部分条文等情况属于源文档解析/切块问题。Trajectory 层不能可靠补全文本，否则会制造训练数据。需要检查原始 PDF、parser 和 chunking 策略，建立法条完整性测试。

## 数据生成注意事项

修复代码后，旧的 `agent_trajectory.jsonl` 不会自动变干净。必须重新生成 trajectory。建议先备份旧结果，再使用：

```powershell
uv run python -m distill.trajectory distill/data/phase5_raw_questions.jsonl --output distill/data/agent_trajectory.jsonl
```

默认会覆盖旧文件，避免重复追加。

## 验收标准

1. 非法律问题不会执行法律 tool。
2. `retrieved_documents` 可以包含候选噪声，但 `citations` 只包含回答实际使用的证据。
3. 一个记录对应一个法条，不能把相邻法条合并成一个 citation 单元。
4. verification 能识别无效/缺失 citation。
5. trajectory 生成失败必须明确报错，不允许静默写入 0 条。
6. 在 Retriever 暴露 score 后，再加入相关性 threshold/rerank benchmark。
