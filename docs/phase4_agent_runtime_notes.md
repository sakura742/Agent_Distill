# Phase 4：LangGraph Agent Runtime + Hybrid Router

## 目标

把 Phase 2/3 的 MCP Tool Service 与 Legal RAG 组织成状态化 Agent Runtime：

```text
START
  ↓
Intent Analysis
  ↓
Task Planning
  ↓
Tool Decision
  ↓
Tool Execution
  ↓
Retrieval
  ↓
Generation
  ↓
Verification
  ├── success → END
  └── failure → Re-plan → Tool Execution
```

## 主要实现

- `agent/runtime/state.py`：统一 `AgentState`，保存问题、领域、意图、计划、工具调用、检索结果、引用、答案、验证状态和 trace。
- `agent/router.py`：Hybrid Router。当前顺序为规则命中 → Embedding 原型相似度 → fallback；后续 Phase 5 可接本地模型作为低置信度兜底。
- `agent/runtime/nodes.py`：Intent Analysis、Task Planning、Tool Decision、Tool Execution、Retrieval、Generation、Verification、Re-plan 节点。
- `agent/runtime/graph.py`：使用 LangGraph `StateGraph` 编排节点和条件边。
- `tests/test_phase4_runtime.py`：使用 Fake Tool Service 验证路由、状态流转和 trace，不依赖真实模型或知识库。

## 与 Phase 2/3 的边界

Phase 4 不重新实现 Chroma、Embedding 或法律 Chunking。Tool Execution 通过 `LegalRetrieverService` 使用 Phase 2/3 的统一 Tool Contract 和 `LegalRetriever`。

Generation 保留可注入的 `answer_generator` 接口；Phase 5 再接 Qwen3.5-2B Serving，不把模型加载逻辑耦合进 Graph Node。

## 当前 Router 设计

```text
Question
   ↓
Rule Router
   │ 命中
   ├──────────────→ Domain
   │ 未命中
   ↓
Embedding Router
   │
   ├──────────────→ Domain
   │ 不可用
   ↓
Fallback
```

当前支持 `labor` / `civil` 两个领域。规则层优先保证常见法律问题的稳定路由，Embedding 层解决规则未覆盖的表达方式；低置信度 LLM 路由预留给后续模型层。

## 验收边界

本阶段代码包含完整的 Graph 和 Mock Tool 测试，但真实法律 PDF、真实 MCP Transport、真实本地模型推理仍需在本地环境进行集成测试。不要在未执行 Benchmark 前填写虚构的路由准确率或 Task Success Rate。
