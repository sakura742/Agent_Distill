# 垂直领域 Agent 能力蒸馏系统

## 项目概览

本项目的核心目标是：**将大模型（DeepSeek / Claude）在法律场景下的 Agent 能力，蒸馏到本地可部署的 Qwen2.5-1.5B 轻量模型上**，同时构建一套完整的法律知识库 RAG 后端，使小模型也能通过工具调用精准检索法律条文，在保护数据隐私的前提下实现垂直领域的 Agent 能力平替。

整个系统由两个模块组成，存在明确的依赖关系：

```
[法律知识库 RAG 模块]  ──提供工具接口──►  [Agent 蒸馏模块]
  ingest.py                                  gen_data.py（教师模型生成数据）
  server.py（MCP服务端）                      train.py（LoRA微调Qwen）
  debug_rag.py                               evaluate.py（验证效果）
                                             agent_pipeline.py（全链路闭环演示）
                                             compare_demo.py（蒸馏前后对比）
```

---

## 目录结构

```
Agent_Distill/
│
├── legal_rag/                        # 模块一：法律知识库（RAG + MCP 服务端）
│   ├── ingest.py                     # PDF 解析、分块、写入向量数据库
│   ├── server.py                     # MCP 服务端，对外暴露检索工具
│   └── debug_rag.py                  # MCP 客户端连通性测试脚本
│
├── distill/                          # 模块二：Agent 能力蒸馏
│   ├── gen_data.py                   # 调用教师模型 API 生成训练数据
│   ├── train.py                      # LoRA 微调 Qwen2.5-1.5B
│   ├── evaluate.py                   # 微调前后工具调用准确率对比评估
│   └── tools_config.json             # 工具定义（供数据生成和推理使用）
│
├── inference/                        # 推理与演示
│   ├── agent_pipeline.py             # 全链路闭环：Qwen决策 → 工具调用 → RAG增强 → 最终回答
│   └── compare_demo.py               # 多场景对比演示：原始模型 vs 微调模型 vs RAG增强
│
├── utils/
│   └── compress.py                   # float32/uint8 向量压缩工具（探索性代码，未接入主流程）
│
├── data/                             # 原始法律文档（需自行准备）
│   ├── labor_law.pdf                 # 《中华人民共和国劳动合同法》
│   └── minfa.pdf                     # 《中华人民共和国民法典》
│
├── chroma_db/                        # 向量数据库持久化目录（ingest.py 自动生成）
├── qwen_mcp_lora_output/             # LoRA 权重输出目录（train.py 自动生成）
├── qwen_merged/                      # 合并后的完整模型（merge_model.py 自动生成）
├── agent_distill_train.jsonl         # 生成的训练数据（gen_data.py 自动生成）
└── merge_model.py                    # 一次性工具：将 LoRA 权重合并进基座模型
```

---

## 模块一：法律知识库 RAG（`legal_rag/`）

这是整个系统的**检索后端**，负责将法律 PDF 文档解析并向量化存储，通过 MCP 协议对外暴露检索工具，供上层 Agent 调用。

### `ingest.py` — 文档解析与向量化入库

**作用**：一次性预处理脚本，将法律 PDF 解析为文本块并写入本地 Chroma 向量数据库。运行一次即可，后续 server.py 直接加载持久化的数据库。

**核心流程**：

1. 使用 `PyPDFLoader` 解析 `data/` 目录下的 `labor_law.pdf` 和 `minfa.pdf`
2. 使用 `RecursiveCharacterTextSplitter` 进行文本分块
   - `chunk_size=500`，`chunk_overlap=50`（保证相邻块之间有语义衔接）
3. 使用 `shibing624/text2vec-base-chinese` 作为 Embedding 模型，将文本块向量化
4. 写入本地 Chroma 向量数据库，持久化至 `chroma_db/` 目录

**使用方式**：
```bash
# 首次使用前必须运行，之后无需重复
python legal_rag/ingest.py
```

---

### `server.py` — MCP 服务端

**作用**：基于 `FastMCP` 搭建的本地 MCP 服务端，对外暴露一个 `search_law` 工具。Agent 调用该工具时，服务端从 Chroma 向量库中检索最相关的法律条文片段并返回。

**关键细节**：

- 启动时加载持久化的 Chroma 向量库（若不存在会抛出异常并提示先运行 `ingest.py`）
- Embedding 模型与 `ingest.py` 保持一致（`text2vec-base-chinese`），否则检索结果会出错
- Retriever 默认返回 top-3 相关文档片段（`search_kwargs={"k": 3}`）
- 暴露的工具只有一个 `search_law(query: str)`，通用检索，不区分民法/劳动法

> **注意**：`server.py` 暴露的工具名是 `search_law`（通用），而 `tools_config.json` 中定义的工具名是 `search_civil_law` 和 `search_labor_law`（分类）。这是一个有意为之的分层设计：蒸馏阶段让小模型学会**意图分类**（判断是民法还是劳动法），实际检索阶段统一路由给同一个向量库。

**使用方式**：
```bash
# 需要保持运行，作为后端服务
python legal_rag/server.py
```

---

### `debug_rag.py` — MCP 连通性测试

**作用**：以 MCP 客户端身份连接本地 `server.py`，验证整条 RAG 链路是否通畅。可以理解为一个集成测试脚本，用于排查 MCP 握手、工具注册、检索召回等环节的问题。

**核心流程**：

1. 通过 `StdioServerParameters` 以子进程方式拉起 `server.py`
2. 建立 MCP 会话并完成初始化握手
3. 列出所有已注册工具，验证 `search_law` 工具存在
4. 发起一次实际检索调用（测试 query：`"定期劳动合同什么时候转为不定期"`）
5. 打印返回的法律原文片段

**特别处理**：针对 Windows 平台 asyncio 的已知兼容问题，手动设置了 `WindowsSelectorEventLoopPolicy`。

**使用方式**：
```bash
# server.py 不需要提前手动启动，debug_rag.py 会自动以子进程拉起
python legal_rag/debug_rag.py
```

---

## 模块二：Agent 能力蒸馏（`distill/`）

这是整个系统的**训练核心**，通过让教师大模型（DeepSeek）自动生成带有思维链（CoT）和工具调用的训练数据，然后用 LoRA 对 Qwen2.5-1.5B 进行监督微调（SFT），使小模型习得在法律场景下进行 Agent 决策的能力。

---

### `tools_config.json` — 工具定义配置

**作用**：以 JSON Schema 格式描述两个法律检索工具的名称、功能和参数，是蒸馏数据生成和推理阶段的**共享契约**。

**包含工具**：

| 工具名 | 作用 | 必填参数 |
|---|---|---|
| `search_civil_law` | 检索民法典（婚姻、财产、合同、侵权等） | `query: str` |
| `search_labor_law` | 检索劳动法/劳动合同法（加班费、裁员、社保等） | `query: str` |

两个工具均支持可选参数 `limit: int`（返回条数，默认 3）。

该文件在 `gen_data.py` 中被读取，注入到教师模型的系统提示词中，确保生成的训练数据与工具接口定义保持一致。

---

### `gen_data.py` — 蒸馏数据生成

**作用**：调用 DeepSeek API（`deepseek-chat` 模型），以其为教师模型，批量生成带有 CoT 思维链和工具调用的训练数据，写入 `agent_distill_train.jsonl`。

**数据生成策略**：

系统提示词要求教师模型按照以下比例生成覆盖不同场景的训练样本：

| 类型 | 比例 | 示例场景 | 对应工具 |
|---|---|---|---|
| 劳动法咨询 | 约 40% | 加班费、违法裁员、未签合同 | `search_labor_law` |
| 民法典咨询 | 约 40% | 借贷纠纷、租房押金、离婚财产 | `search_civil_law` |
| 负样本（闲聊） | 约 20% | "你好"、宏观闲聊 | `null`（不调用工具） |

负样本的设计至关重要：防止小模型对所有输入都无脑触发工具调用，学会在不需要时**不调用**工具。

**生成数据格式（写入 JSONL）**：

```json
{
  "instruction": "你是一个熟练对接后端 MCP 法律知识库服务器的智能助手...",
  "input": "用户的法律问题",
  "output": "思考：Agent 的 CoT 推理过程\n行动：调用法律 MCP 服务器工具 search_labor_law，参数为 {...}"
}
```

**运行规模**：脚本循环 10 批次，每批 20 条，共生成 **200 条**训练数据（保存为 `agent_distill_train.jsonl`，约 154 KB）。

**使用方式**：
```bash
# 需要先配置 DeepSeek API Key（脚本第 6 行）
python distill/gen_data.py
```

---

### `train.py` — LoRA 微调训练

**作用**：以 `agent_distill_train.jsonl` 为训练数据，对 Qwen2.5-1.5B 进行 LoRA 监督微调（SFT），使其习得在法律场景下识别意图、输出工具调用的 Agent 能力。

**关键训练配置**：

| 配置项 | 值 | 说明 |
|---|---|---|
| 基座模型 | Qwen2.5-1.5B | 本地部署，离线推理 |
| 量化 | INT8（`BitsAndBytesConfig`） | 降低显存占用 |
| LoRA rank | `r=8`, `alpha=16` | 内存占用约 6G，单卡可训 |
| LoRA 注入位置 | q/k/v/o_proj + gate/up/down_proj | 覆盖注意力层和 FFN 层 |
| Dropout | 0.1 | 防止过拟合 |
| 训练轮数 | 3 epochs | |
| Batch size | 1（梯度累积 4 步） | 等效 batch size = 4 |
| 学习率 | 2e-4 | |
| 优化器 | `paged_adamw_8bit` | 配合 INT8 量化节省显存 |
| 精度 | bf16 | |
| 最大序列长度 | 1024 | |

**Prompt 格式**（ChatML 格式，与 Qwen 原生格式对齐）：

```
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

训练完成后，LoRA 权重保存至 `qwen_mcp_lora_output/`。

**使用方式**：
```bash
# 需要先运行 gen_data.py 生成训练数据
python distill/train.py
```

---

### `evaluate.py` — 评估脚本

**作用**：在 5 道标准测试题上，对比**原始 Qwen2.5-1.5B** 与**微调后 Qwen2.5-1.5B** 的工具调用准确率，验证蒸馏效果。

**评估方式**：

- 判断模型输出中是否包含**期望的工具名**（字符串匹配，不做 JSON 解析）
- 每道题提供标准 CoT 和标准工具调用，便于观察模型输出的质量

**测试题覆盖场景**：

| 题号 | 场景 | 期望工具 |
|---|---|---|
| 1 | 口头辞退、违法解除劳动合同 | `search_labor_law` |
| 2 | 民间借贷 5 万，债务追偿 | `search_civil_law` |
| 3 | 查询民法典第 1079 条原文 | `get_law_article` |
| 4 | 咨询起诉前房东的诉讼流程 | `legal_advice` |
| 5 | 公司强制签自愿放弃加班费协议 | `search_labor_law` |

> **说明**：题 3、4 中的 `get_law_article` 和 `legal_advice` 工具在 `tools_config.json` 中未定义，仅在 evaluate.py 的工具描述文本中出现，用于测试模型泛化能力。

**使用方式**：
```bash
python distill/evaluate.py
```

---

## 推理与演示（`inference/`）

### `agent_pipeline.py` — 全链路闭环演示

**作用**：演示完整的三阶段 Agent 执行流程，使用微调后的 Qwen 模型作为前端决策器，对接本地法律数据库（模拟 MCP 调用），最终输出有法律依据的回答。

**三阶段流程**：

```
阶段一：Agent 决策
  微调后的 Qwen 接收用户问题
  → 输出 CoT 思维链 + 工具调用 JSON
  （例：{"tool": {"name": "search_labor_law", "params": {"keywords": [...]}}}）

阶段二：工具执行
  Python 解析模型输出的 JSON
  → 调用 call_local_mcp_database() 模拟 MCP 检索
  → 返回真实法条文本（《劳动合同法》第 47/85/87 条等）

阶段三：RAG 增强生成
  将检索到的法条注入新的 Prompt
  → 再次调用 Qwen 生成最终的律师级回答
```

> **注意**：当前 `call_local_mcp_database()` 是基于关键词匹配的模拟实现，实际部署时应替换为真正调用 `server.py` 的 MCP 客户端调用。

**使用方式**：
```bash
python inference/agent_pipeline.py
```

---

### `compare_demo.py` — 蒸馏效果对比演示

**作用**：在 5 个典型法律场景下，并排输出三列结果，直观展示蒸馏效果：

```
原始 Qwen-1.5B 的输出          →  通常无法输出结构化工具调用
微调后 Qwen 的 Agent 决策输出  →  正确输出思维链 + 工具调用意图
对接本地法律库后的最终回答      →  结合真实法条给出有据可查的回答
```

**测试场景覆盖**：

| 场景 | 标签 |
|---|---|
| 突然被踢出工作群、扣工资 | 劳动纠纷 - 违法辞退 |
| 被迫签自愿放弃加班费协议 | 劳动纠纷 - 加班费 |
| 朋友借款 8 万联系不上 | 民事纠纷 - 借贷 |
| 搬家后房东扣押金 | 民事纠纷 - 租房 |
| 查询劳动合同法第 47 条原文 | 条文精确查询 |

**使用方式**：
```bash
python inference/compare_demo.py
```

---

## 工具脚本

### `merge_model.py` — LoRA 权重合并

**作用**：将训练好的 LoRA Adapter 权重合并进基座模型，输出一个完整的标准 HuggingFace 模型，便于部署和分发（无需再依赖 PEFT 库加载）。

合并在 CPU 上进行以节省显存，合并后保存至 `qwen_merged/`。

**使用时机**：在 `train.py` 完成训练后，需要导出完整模型时运行。

```bash
python merge_model.py
```

---

### `compress.py` — 向量压缩工具（未接入）

**作用**：实现 float32 向量与 uint8 向量之间的互转压缩，使用 min-max 归一化，保留最大值的精确位置（强制映射到 255）。

属于探索性代码，**未在当前项目主流程中使用**。原始设想可能是用于压缩 Chroma 向量数据库中存储的 embedding，以减少磁盘占用。

---

## 运行顺序

首次完整运行时，按以下顺序执行：

```
1. python legal_rag/ingest.py           # 解析法律 PDF，建立向量数据库（一次性）
2. python legal_rag/debug_rag.py        # 验证 RAG 链路正常（可选）
3. python distill/gen_data.py           # 生成训练数据（需要 DeepSeek API Key）
4. python distill/train.py              # LoRA 微调（需要 GPU，约 6G 显存）
5. python merge_model.py                # 合并 LoRA 权重（可选，按需导出）
6. python distill/evaluate.py           # 评估微调效果
7. python inference/compare_demo.py     # 查看完整对比效果
```

---

## 技术栈

| 组件 | 使用方案 |
|---|---|
| 基座模型 | Qwen2.5-1.5B（本地部署） |
| 教师模型 | DeepSeek V3（`deepseek-chat`，API 调用） |
| 微调方法 | LoRA（通过 PEFT + TRL SFTTrainer） |
| 向量数据库 | Chroma（本地持久化） |
| Embedding | `shibing624/text2vec-base-chinese` |
| RAG 框架 | LangChain |
| Agent 协议 | MCP（Model Context Protocol，FastMCP） |
| PDF 解析 | LangChain PyPDFLoader |
| 推理框架 | HuggingFace Transformers |
