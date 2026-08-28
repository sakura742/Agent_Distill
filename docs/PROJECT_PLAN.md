# Agent_Distill → Legal Agent Platform 项目计划

| 阶段 | 名称 | 状态 |
|---|---|---|
| 0 | 代码审计 | ✅ |
| 0.5 | 目标态规划 | ✅ |
| 1 | 模块化骨架 + 配置/日志/异常 | ✅ |
| 2 | MCP Tool Service + 工具契约 | ✅ |
| 3 | RAG 分域 + Metadata / Rerank | ✅ |
| 4 | LangGraph Runtime + Hybrid Router | ✅ |
| 5 | Qwen3.5-4B + Trajectory 蒸馏 | ✅ |
| 6 | Benchmark / Evaluation 体系化 | 🔄 进行中 |
| 7 | 多轮对话 + Web Dashboard + Docker | ⬜ |

## Phase 6

主实验固定为 **Qwen3.5-4B Raw vs Qwen3.5-4B LoRA**。

当前 Benchmark：**40 个法律场景**，覆盖 Routing、Retrieval、Tool Calling、Workflow、Answer、Citation。

评估覆盖：知识库检索 Recall@5 / MRR、工具选择准确率、工具参数准确率、Workflow / Task Success Rate、最终回答 F1、Citation Precision / Recall / Exact Accuracy、端到端延迟及长链路中断率。

已实现：
- `evaluation/benchmark.py` / `metrics.py`：统一指标与数据评估。
- `evaluation/run_qwen35_benchmark.py`：真实 Qwen3.5-4B Raw / LoRA Runtime benchmark。
- `evaluation/experiment.py` / `compare_models.py`：paired comparison。
- `evaluation/error_analysis.py`：错误类型统计与困难样本导出。
- `evaluation/iteration.py`：评估 → 错误分析 → hard examples → 重训数据的闭环入口，并自动计算错误率相对下降。
- `evaluation/experiment_config.json`：实验控制变量与闭环定义。
- `tests/test_phase6_evaluation.py`：核心评估指标回归测试。

### 简历指标口径

“长链条任务中断率/错误率降低 35%”必须由同一 Benchmark 上 Raw 与 LoRA 的实际运行结果计算：

`(Raw interruption_error_rate - LoRA interruption_error_rate) / Raw interruption_error_rate`

未达到 0.35 时不得写成 35%，使用实际测量值。

### 当前进度

- 6.1 Benchmark schema + deterministic metrics：✅
- 6.2 Raw / LoRA paired runner：✅
- 6.3 Error taxonomy + hard-example mining：✅
- 6.4 Benchmark 扩充至 40 cases + 回归测试：✅
- 6.5 本地 Qwen3.5-4B Raw / LoRA 全量运行：⏳ 需要本地 checkpoint / RAG 环境执行
- 6.6 根据真实错误样本补充训练集并重新 LoRA：⏳

## 下一步

完成 6.5 → 6.6；拿到真实结果后决定是否达到简历中的 35% 指标，再进入 Phase 7。
