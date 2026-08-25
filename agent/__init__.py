"""agent/ —— Agent 推理层（原 inference/ 目录）。

包含：
- ``inference_core``：Web 服务实际使用的主推理入口（三阶段：工具决策 -> 检索 -> 生成）。
- ``agent_pipeline``：旧的 MCP 全链路推理入口，保留用于演示 MCP 协议闭环。

Phase 1 只做搬迁 + import 修复 + 配置/日志接入，未改变任何推理算法逻辑。
"""
