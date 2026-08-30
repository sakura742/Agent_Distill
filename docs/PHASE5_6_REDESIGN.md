# Phase 5/6 重构说明

## 核心问题
当前 Phase 5 把 domain、intent、plan、tool、evidence、answer 全部拼成 assistant JSON 做 SFT；Phase 6 却用 `question + retrieved evidence -> natural-language answer`。训练目标与推理目标不一致。

同时，当前 LangGraph 中 routing、planning、tool selection 主要由 Python 固定逻辑完成，Qwen 没有真正承担 Agent decision，因此 Raw 与 LoRA 的很多 end-to-end 指标天然相同。

## 重构目标
保留 LangGraph、MCP、RAG 和 Benchmark，不推倒重来，将模型职责拆清：

```text
Question -> Qwen Decision LoRA -> domain/intent/tool/arguments
                                      |
                                      v
                              LangGraph + MCP/RAG
                                      |
                                      v
                                  evidence
                                      |
                                      v
                         Qwen Answer LoRA -> answer
```

## Phase 5 新方案
### 1. Agent Decision SFT
输入：question。
输出只包含模型真正需要决定的结构化 action：domain、intent、tool name、tool arguments。不要训练 evidence、answer、trace。

### 2. Answer SFT
输入：question + evidence；输出 Teacher 的最终法律建议。线上 generation 必须使用同样的消息结构。

两个目标使用独立 LoRA adapter，从同一个 Qwen3.5-4B base model 分别训练，避免 decision objective 污染 answer objective。

## 实施步骤
1. 冻结当前 Raw/LoRA benchmark JSON，作为重构前 baseline，不删除旧 adapter。
2. 重新运行 teacher trajectory，得到 observable supervision record。
3. 从 trajectory 生成 decision JSONL 和 answer JSONL，并按 question 去重。
4. 训练 Decision LoRA，输出独立目录。
5. 从 base model 重新训练 Answer LoRA，输出独立目录。
6. 修改 Runtime 的 `tool_decision`，允许注入 Decision LoRA；JSON 解析失败时保留 deterministic fallback。
7. generation 保持 `question + evidence -> answer`，与 Answer SFT 完全一致。
8. Phase 6 分别评估 routing、tool selection、tool arguments、retrieval Recall@5/MRR、workflow、answer quality、citation grounding 和 end-to-end success。

## 验收标准
- Decision LoRA 输出能够被 Runtime 解析，并实际决定 tool。
- Answer LoRA 训练 prompt 与线上 prompt 一致。
- Raw/LoRA 使用相同 benchmark、retrieval 和 decoding 参数。
- benchmark 与训练数据严格分离。
- 不通过修改 evaluator 人为制造 LoRA 提升。

## 评估说明
Token-overlap F1 只能作为 lexical regression 指标，不能单独代表法律答案正确性。`error_analysis` 从空字符串二元判断改为连续 answer quality 是独立的评估修复，不是 LoRA 质量问题的根因。

## 原则
不训练隐藏 chain-of-thought；先解决 objective alignment，再调 LoRA 超参数；先建立可解释 baseline，再扩大训练数据。
