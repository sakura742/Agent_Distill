# Agent_Distill 改进计划（第二版）

## 项目现状

两个模块：
- **legal_rag/**：法律 PDF → Chroma 向量库 → MCP 服务端（server.py）
- **distill/**：DeepSeek 生成训练数据 → LoRA 微调 Qwen2.5-1.5B → 评估

**当前状态**：端到端链路已闭合（MCP 版本），evaluate.py 已重构可信。
**下一阶段目标**：去除推理对 MCP 的依赖、建立 Web 评估平台、支持多轮对话。

---

## 目标一：新推理入口（inference_core.py）

**背景**：当前 `agent_pipeline.py` 通过 MCP 协议调用 `server.py` 检索法条，链路复杂、子进程管理麻烦、不适合嵌入 Web 服务。新建 `inference/inference_core.py`，直接调用 Chroma 向量库，`agent_pipeline.py` 保留不动。

### 新旧两种推理模式对比

| | agent_pipeline.py（旧） | inference_core.py（新） |
|---|---|---|
| 检索方式 | MCP 协议 → 子进程 server.py | 直接 retriever.invoke() |
| 依赖 | mcp, asyncio, stdio_client | langchain, chromadb |
| 适合场景 | 演示 MCP 完整链路 | Web 服务、多轮对话 |
| 保留 | ✓ 保留 | ✓ 新建 |

### inference_core.py 的结构

三个函数，对外暴露清晰接口，供 Web 服务直接调用：

```
load_models()                              # 启动时调用一次，加载 Qwen + LoRA + Chroma retriever
rag_search(query) -> str                   # 直接调用 retriever.invoke()，返回法条文本
run_inference(user_query, history) -> dict # 核心推理，支持传入对话历史，返回 answer + reasoning
```

**推理流程**（与 agent_pipeline 一致，只替换检索层）：

```
阶段一：Qwen 分析问题 → 输出 JSON（thought + tool name + query）
阶段二：解析 JSON → 直接 retriever.invoke(query)，不走 MCP
阶段三：法条 + 对话历史 + 用户问题 → Qwen 生成最终回答
```

**注意事项**：
- Chroma 和 Embedding 模型在 `load_models()` 里一次性加载，不要每次请求都重新加载
- `db_result` 截断到 600 字符再拼入 prompt（防止超出 Qwen 2048 token 上下文窗口）
- 阶段一 `max_new_tokens=256`（实际运行发现 128 不够，JSON 被截断），阶段三 `max_new_tokens=512`

---

## 目标二：Web 评估平台（含多轮对话）

### 整体定位

供法律专业同学评估两个模型的回答质量，不需要打分，只需要并排看到两列回答，同时能看到推理过程（thought + 工具调用）。支持多轮对话，即同一个案件可以追问细节。

### 技术方案

**后端**：FastAPI，单文件 `web/app.py`

**前端**：单个 HTML 文件 `web/index.html`，不引入前端框架，纯 JS + CSS，够用即可

**多轮记忆**：LangChain `ConversationalRetrievalChain` + `ConversationBufferWindowMemory`
- 每个会话维护一个独立的 memory 对象，存在后端内存里（本地够用，不需要数据库）
- 窗口大小设为 `k=3`（保留最近 3 轮），避免历史过长撑爆 Qwen 的上下文窗口
- 新建会话时清空 memory，前端提供"新建对话"按钮

### 两列模型说明

| | 左列：原始 Qwen | 右列：微调后 Qwen |
|---|---|---|
| 模型 | Qwen2.5-1.5B 原始权重 | Qwen2.5-1.5B + LoRA |
| 知识库 | 无 | 有（Chroma 向量库） |
| System Prompt | 精心设计的法律顾问 prompt，让模型在无法条支撑下尽可能表现好 | 与 inference_core 一致 |
| 推理过程展示 | 不展示（无工具调用） | 展示 thought + 工具名 + 检索到的法条片段 |

> 原始模型的 system prompt 目标：公平对比。不能让原始模型输在 prompt 工程上，要让它在没有法条的情况下发挥出最好水平，这样右列的提升才真正归因于知识库和蒸馏。

### 页面布局

```
┌─────────────────────────────────────────────────────┐
│  法律咨询评估平台              [新建对话]            │
├──────────────────────┬──────────────────────────────┤
│   原始 Qwen-1.5B     │   微调 Qwen-1.5B + 知识库    │
│   （无知识库）        │                              │
│                      │   [推理过程 ▼ 可折叠]         │
│   回答内容           │   thought: ...               │
│                      │   工具: search_labor_law      │
│                      │   法条: ...                  │
│                      │   ──────────────             │
│                      │   回答内容                   │
├──────────────────────┴──────────────────────────────┤
│  [历史对话记录，可折叠]                              │
├──────────────────────────────────────────────────────┤
│  输入框                              [发送]          │
└─────────────────────────────────────────────────────┘
```

### API 设计

```
POST /chat
  请求：{ "session_id": "xxx", "message": "用户问题" }
  返回：{
    "raw_answer": "原始模型回答",
    "tuned_answer": "微调模型回答",
    "reasoning": {
      "thought": "...",
      "tool": "search_labor_law",
      "law_snippet": "检索到的法条前300字"
    }
  }

POST /session/new
  返回：{ "session_id": "uuid" }

GET /session/{session_id}/history
  返回：该会话的对话历史列表
```

### 文件结构

```
web/
├── app.py        # FastAPI 后端，调用 inference_core.py
└── index.html    # 前端页面，单文件
```

启动方式：
```bash
# 确保已运行 ingest.py 建库
uvicorn web.app:app --host 0.0.0.0 --port 8000
# 浏览器访问 http://localhost:8000
```

---

## 目标三：多轮对话性能测试

**说明**：多轮能力已通过 Web 平台的 `ConversationalRetrievalChain` 实现，本目标专注于**系统性测试**，验证模型在多轮场景下的表现，独立于 Web 服务运行。

### 测试脚本：`distill/evaluate_multiturn.py`

区别于现有的 `evaluate.py`（单轮、工具调用准确率），本脚本测试：

1. **上下文记忆**：模型能否在第 2、3 轮正确引用第 1 轮的信息
2. **追问处理**：对同一问题的追问能给出更细节的回答，而不是重复第一轮
3. **话题切换**：上下文完全切换后，模型不应把旧上下文带入新问题

### 测试用例设计（3 组，每组 3 轮）

**组 1：劳动纠纷追问**
```
轮1：我在公司干了3年，上周被口头辞退，有赔偿吗？
轮2：赔偿金具体怎么计算？（期望：引用轮1的"3年"）
轮3：如果公司不赔怎么办？（期望：给出维权步骤）
```

**组 2：民事借贷追问**
```
轮1：借给朋友5万块一直不还，我能起诉吗？
轮2：诉讼时效是多久？（期望：结合轮1场景回答）
轮3：如果超过诉讼时效了还有救吗？
```

**组 3：话题切换测试**
```
轮1：公司强制要求签放弃加班费协议，合法吗？
轮2：好的谢谢。（期望：正常结束，不调用工具）
轮3：房东装修扰民我该怎么办？（期望：切换到民法，不带入劳动法上下文）
```

### 评估指标

每组每轮人工判断，记录以下三项：

| 指标 | 说明 |
|---|---|
| 上下文引用正确 | 追问时是否正确引用了前轮关键信息 |
| 工具调用合理 | 该轮是否正确判断了需要/不需要调用工具 |
| 回答未重复 | 追问的回答是否比上一轮更具体，没有原文重复 |

---

## 后续可选方向

### DATA-1：训练数据加入拒绝采样

在 `gen_data.py` 写入前校验：JSON 能解析、工具名在白名单内、thought 字段不少于 20 字。当前 200 条数据未过滤，有噪声样本影响训练质量。

### MODEL-1：训练数据 output 格式改为严格 JSON

当前训练数据 output 是中文自然语言混合格式，模型学不到规范的 JSON 输出习惯。改成纯 JSON 格式后重新生成数据并重训，推理时可直接 `json.loads()` 解析，去掉正则兜底。这是从根本上提升输出稳定性的方法，工作量较大（改 gen_data.py + 重新生成 + 重训）。

### MODEL-2：DPO 偏好对齐

SFT 之后用 TRL 的 `DPOTrainer` 再训一轮，用正确工具调用作为 chosen、错误调用作为 rejected，预期准确率再提升 5-10 个点。

---

## 已完成事项（简记）

- BUG-1：server.py 拆分为 search_civil_law / search_labor_law，工具名与训练数据对齐
- BUG-2：agent_pipeline.py 替换为真实 MCP 调用，端到端链路闭合
- BUG-3 + EVAL-1：evaluate.py 完全重构，测试题全部使用已实现工具，评估升级为 JSON 精确匹配
- 推理稳定性：修复 torch_dtype 警告、上下文溢出导致回答错位、MCP 失败时注入废话等问题
