# Phase 5：Agent Distillation

## 目标

把原来的“问题 → 工具调用字符串”训练方式升级为结构化 Agent Trajectory，并让教师生成、数据校验、SFT 转换、LoRA 训练和推理适配彼此解耦。

## 已完成

- `distill/trajectory_schema.py`：统一 `AgentTrajectory / AgentStep / ToolCall` 契约。
- `distill/teacher_pipeline.py`：DeepSeek 教师模型结构化生成轨迹。
- `distill/validate_dataset.py`：批量校验 JSONL 轨迹，报告缺失字段和类型错误。
- `distill/convert_dataset.py`：将 canonical trajectory 转换为现有 SFT JSONL 格式。
- `distill/model_adapter.py`：隔离 Transformers/PEFT 加载细节，为 LangGraph 提供轻量本地模型接口。
- `tests/test_phase5_distillation.py`：覆盖轨迹 schema 和数据转换。

## 数据设计

每条轨迹包含：

- `user_query`
- `intent`
- `domain`
- `steps[]`
  - `action`
  - `tool_call`
  - `observation`
- `final_answer`
- `metadata`

训练数据不再把工具调用埋在一段自然语言中，而是先保留结构化轨迹，再转换为 SFT 文本。这样后续可以分别评估 intent、tool selection、argument accuracy 和 task execution。

## 与 Phase 4 的关系

Phase 4 的 LangGraph Runtime 是执行基准；Phase 5 的教师轨迹应逐步由真实 Runtime 产生，而不是仅依赖教师模型凭空描述 observation。当前 `teacher_pipeline.py` 负责第一阶段结构化教师标注，下一步应增加 Runtime-backed trajectory collector：教师决定工具 → 实际调用 MCP → 保存 observation → 再请求教师决定下一步。

## 模型

当前训练脚本仍保留历史 Qwen2.5-1.5B LoRA 参数作为兼容入口。Phase 5 的本地模型路径由 `AGENT_DISTILL_BASE_MODEL_PATH` 控制，因此切换到 Qwen3.5-2B 不需要修改代码；在实际训练前应确认当前 Transformers/PEFT/TRL 版本对目标 Qwen3.5 checkpoint 的架构支持，并用小规模 smoke test 验证显存与 tokenizer/chat template。

## 下一步

1. Runtime-backed Teacher Trajectory Collector。
2. Tool schema 严格校验与非法工具调用拒绝。
3. Hard-example mining：从 Phase 6 评估失败样本回流训练集。
4. LoRA smoke training（少量样本）后再进行完整训练。
5. 将微调模型接入 Phase 4 LangGraph 的 `intent_analysis` / `tool_execution` 节点，比较 Base vs LoRA。
