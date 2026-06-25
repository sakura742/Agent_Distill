# Agent_Distill 改进计划

## 项目现状（必读背景）

这是一个将大模型 Agent 能力蒸馏到 Qwen2.5-1.5B 的项目，包含两个模块：
- **legal_rag/**：法律 PDF → Chroma 向量库 → MCP 服务端（`server.py`）
- **distill/**：DeepSeek 生成训练数据 → LoRA 微调 Qwen → 评估

**当前最大问题：两个模块各自跑通，但端到端链路是断的。**

---

## Bug 修复（必须先做）

### BUG-1：工具名不一致导致推理时调用失败

**问题**：`distill/tools_config.json` 定义了两个工具名 `search_civil_law` / `search_labor_law`，小模型训练时学的就是这两个名字。但 `legal_rag/server.py` 实际暴露的工具名是 `search_law`（只有一个）。推理时小模型输出的工具名在 MCP 服务端根本找不到，调用必然失败。

**修复方案（二选一）**：
- 方案 A（推荐）：`server.py` 拆成两个工具 `search_civil_law` 和 `search_labor_law`，内部都调用同一个 retriever，只是入口分开。
- 方案 B：`tools_config.json` 改成单工具 `search_law`，重新生成训练数据，重新训练。

---

### BUG-2：`agent_pipeline.py` 的工具调用是假的

**问题**：`inference/agent_pipeline.py` 是演示端到端闭环的脚本，但其中 `call_local_mcp_database()` 函数是用 if/else 关键词匹配硬编码的模拟实现，并没有真正调用 `legal_rag/server.py`。`debug_rag.py` 已经验证 MCP 链路可以跑通，但 `agent_pipeline.py` 完全没有用上它。

**修复方案**：将 `agent_pipeline.py` 中的 `call_local_mcp_database()` 替换为真正的 MCP 客户端调用，参考 `debug_rag.py` 中的 `stdio_client` + `ClientSession` 写法，通过子进程拉起 `server.py` 并调用 `search_law` 工具。

---

### BUG-3：`evaluate.py` 中有两个未实现的工具，评估结果失真

**问题**：`evaluate.py` 的 5 道测试题中，题 3 期望工具 `get_law_article`，题 4 期望工具 `legal_advice`。这两个工具既没有在 `tools_config.json` 中定义，`server.py` 也没有实现，小模型从未见过它们。这两道题注定是错的，导致评估准确率虚低。

**修复方案（二选一）**：
- 方案 A：在 `server.py` 和 `tools_config.json` 中补全这两个工具，并补充对应训练数据后重训。
- 方案 B（快速）：将这两道题改为 `search_civil_law` / `search_labor_law` 场景，保持评估集与训练数据的工具集一致。

---

## 数据质量改进

### DATA-1：加入拒绝采样，过滤低质量训练样本

**问题**：`gen_data.py` 生成的 200 条数据全部直接写入训练集，没有任何过滤。DeepSeek 偶尔会输出 JSON 格式错误、工具名拼写错误、thought 与 tool_call 逻辑矛盾的样本，这些噪声数据会直接损害训练效果。

**改进方案**：在 `gen_data.py` 写入前加校验逻辑：
1. JSON 能否正常解析
2. `tool_call.name` 是否在合法工具名列表中（`search_civil_law` / `search_labor_law` / `null`）
3. `thought` 字段长度是否合理（过短说明推理质量差）

只有通过校验的样本才写入 `.jsonl`。

---

### DATA-2：增加负样本比例

**问题**：当前训练数据负样本（不调用工具的闲聊场景）占比约 20%，偏低。实际使用中用户经常发送无关问题，模型容易对所有输入都触发工具调用（过度调用）。

**改进方案**：将负样本比例提高到 30%~35%，并丰富负样本类型，不只是"你好"这类明显闲聊，还要包括"法律相关但不需要检索条文"的场景（如"律师一般收费多少"、"打官司要多久"）。在 `gen_data.py` 的 SYSTEM_PROMPT 中更新比例要求即可。

---

## 评估体系改进

### EVAL-1：将字符串匹配升级为结构化评估

**问题**：`evaluate.py` 当前只检查模型输出中是否**包含**期望工具名（`item["expected_tool"] in result`），这个方式太宽松，无法区分"正确调用"和"输出中恰好出现了工具名字符串"。

**改进方案**：增加三层评估指标：
1. **工具名正确**：输出 JSON 中 `tool.name` 字段与期望工具名完全匹配
2. **JSON 格式合法**：输出能被 `json.loads()` 正常解析
3. **参数质量评估**：`query` 参数中是否包含核心关键词（可用简单关键词列表做近似判断）

---

## 模型能力提升（进阶）

### MODEL-1：将训练数据 output 格式改为严格 JSON

**问题**：当前训练数据的 `output` 字段是混合中文自然语言的格式（`"思考：...\n行动：调用工具..."`），推理时需要用正则或字符串解析，不稳定且容易出错。

**改进方案**：将 output 格式改为纯 JSON 结构：
```json
{
  "thought": "用户问题涉及违法解除劳动合同...",
  "tool": {
    "name": "search_labor_law",
    "arguments": {"query": "违法解除劳动合同 赔偿"}
  }
}
```
同步修改 `gen_data.py` 的 SYSTEM_PROMPT 输出格式要求，以及 `train.py` 的 `formatting_prompts_func`。推理时直接 `json.loads()` 解析输出，不需要任何正则处理。

---

### MODEL-2：SFT 之后做一轮 DPO 偏好对齐

**背景**：DPO（Direct Preference Optimization）通过对比正确样本（chosen）和错误样本（rejected）进一步对齐模型行为，在 SFT 基础上通常能再提升 5-10 个百分点的准确率。

**实施方案**：
1. 用当前微调模型对训练集问题做推理，收集输出
2. 工具名正确的输出作为 `chosen`，工具名错误或格式错误的输出作为 `rejected`
3. 使用 TRL 的 `DPOTrainer` 在 LoRA 基础上继续训练一轮

---

## 技术栈瘦身（可选）

### TECH-1：移除 LangChain 依赖

**问题**：`ingest.py` 引入了 LangChain，但实际只用了 `PyPDFLoader` 和 `RecursiveCharacterTextSplitter` 两个组件，引入整个框架依赖过重，且 LangChain 版本迭代频繁，容易出现兼容性问题。

**替代方案**：
- PDF 解析：改用 `pymupdf`（`fitz`），速度更快，依赖更轻
- 文本分块：手写 `chunk_text(text, size=500, overlap=50)` 函数，十行以内

`server.py` 中的 LangChain Chroma 封装可保留，或直接用 `chromadb` 原生 Python 客户端替代。

---

## 优先级排序

| 优先级 | 任务 | 原因 |
|---|---|---|
| P0 | BUG-1 工具名统一 | 不修会导致推理完全失败 |
| P0 | BUG-2 打通真实 MCP 调用 | 项目核心链路尚未闭合 |
| P1 | BUG-3 修复评估集 | 评估结果当前不可信 |
| P1 | DATA-1 拒绝采样 | 低成本提升训练数据质量 |
| P2 | EVAL-1 结构化评估 | 让评估指标真正可信 |
| P2 | DATA-2 增加负样本 | 减少过度调用问题 |
| P3 | MODEL-1 JSON 输出格式 | 提升推理稳定性，需重新生成数据+重训 |
| P3 | MODEL-2 DPO | 进阶提升，工作量较大 |
| P4 | TECH-1 移除 LangChain | 工程整洁性，功能不受影响 |
