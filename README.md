# Agent_Distill 项目文档

## 项目简介

将大模型（DeepSeek）在法律场景下的 Agent 能力蒸馏到本地可部署的 Qwen2.5-1.5B，
结合本地法律知识库（RAG），实现在保护数据隐私前提下的垂直领域 Agent 能力平替。
配套 Web 评估平台供法律专业人员对比原始模型与蒸馏模型的回答质量。

---

## 目录结构

```
Agent_Distill/
│
├── data/                          # 原始法律文档
│   ├── labor_law.pdf              # 《中华人民共和国劳动合同法》
│   └── minfa.pdf                  # 《中华人民共和国民法典》
│
├── legal_rag/                     # 模块一：法律知识库
│   ├── ingest.py                  # PDF 解析、分块、写入 Chroma（一次性运行）
│   ├── server.py                  # MCP 服务端，暴露 search_civil_law / search_labor_law
│   ├── debug_rag.py               # MCP 客户端连通性测试
│   └── chroma_db/                 # 向量数据库持久化目录（自动生成）
│
├── distill/                       # 模块二：Agent 能力蒸馏
│   ├── tools_config.json          # 工具定义（search_civil_law / search_labor_law）
│   ├── gen_data.py                # 调用 DeepSeek API 生成 CoT + 工具调用训练数据
│   ├── train.py                   # LoRA 微调 Qwen2.5-1.5B
│   └── evaluate.py                # 微调前后工具调用准确率对比（JSON 精确匹配）
│
├── inference/                     # 推理层
│   ├── inference_core.py          # 主推理入口：直接调 Chroma，串行双模型，供 Web 调用
│   ├── agent_pipeline.py          # 旧推理入口：MCP 协议版本，保留备用
│   ├── direct_rag.py              # 轻量检索工具：绕过 LangChain，直接用 chromadb 原生客户端检索
│   └── test_mcp_server.py         # 独立诊断工具：压测 legal_rag/server.py MCP 管道通信
│
├── web/                           # Web 评估平台
│   ├── app.py                     # FastAPI 后端：会话管理 + 调用 inference_core
│   └── index.html                 # 前端：双列对比 + 多轮对话 + 推理过程展示
│
├── untils/
│   └── compress.py                # float32/uint8 向量压缩与解压工具（探索性代码，未接入主流程）
│
├── version/                       # 历史文档归档
│   ├── Agent_Distill_ProjectDoc.md # 初始项目设计文档：模块划分、依赖关系、运行方式
│   ├── improvement_plan.md         # 改进计划 v1：分析端到端链路断裂问题，规划 Web 评估平台
│   └── improvement_plan_v2.md      # 改进计划 v2：拆分为三个目标，细化实施步骤与注意事项
│
├── tree.py                        # 辅助工具：递归打印项目目录树结构，自动排除模型/虚拟环境目录
├── merge_model.py                 # 一次性工具：在 CPU 上将 LoRA 权重合并进基座，导出完整模型到 qwen_merged/
├── agent_distill_train.jsonl      # 生成的训练数据（gen_data.py 自动生成，200 条）
├── qwen_mcp_lora_output/          # LoRA 权重输出目录（train.py 自动生成）
├── qwen_merged/                   # 合并后的完整模型（merge_model.py 自动生成）
├── .gitignore                     # Git 忽略规则：排除模型文件、向量库、虚拟环境
├── .python-version                # uv 管理的 Python 版本声明
├── pyproject.toml                 # 项目元数据与依赖声明（uv 包管理器）
├── uv.lock                        # uv 依赖锁定文件
└── README.md
```

---

## 各模块说明

### legal_rag/（法律知识库）

| 文件 | 作用 |
|---|---|
| `ingest.py` | 用 pymupdf 解析两本法律 PDF，手写分块（500字/50重叠），写入 Chroma 向量库。只需运行一次。|
| `server.py` | 基于 FastMCP 的 MCP 服务端。加载持久化 Chroma，暴露 `search_civil_law` 和 `search_labor_law` 两个工具，内部共用同一 retriever。|
| `debug_rag.py` | 以 MCP 客户端身份拉起 server.py 子进程，验证握手、工具注册、检索召回全链路是否正常。|

Embedding 模型：`shibing624/text2vec-base-chinese`，运行在 CPU。

---

### distill/（Agent 能力蒸馏）

| 文件 | 作用 |
|---|---|
| `tools_config.json` | 定义两个工具的名称、描述、参数 schema，是数据生成和推理的共享契约。|
| `gen_data.py` | 调用 DeepSeek API（`deepseek-chat`）批量生成训练数据：劳动法场景40%、民法场景40%、负样本20%，循环10批共200条，写入 `agent_distill_train.jsonl`。|
| `train.py` | INT8 量化加载 Qwen2.5-1.5B，注入 LoRA（r=8，覆盖全部注意力层和 FFN），用 TRL SFTTrainer 训练3轮，约6G显存，LoRA 权重保存至 `qwen_mcp_lora_output/`。|
| `evaluate.py` | 5道测试题（劳动法×2、民法×2、负样本×1），对比原始模型和微调模型的工具调用准确率。主评估为 JSON 字段精确匹配，降级为正则匹配，分别统计两种命中数。|

---

### inference/（推理层）

| 文件 | 作用 |
|---|---|
| `inference_core.py` | **主推理入口**。启动时只加载 tokenizer + Chroma retriever；每次请求串行加载微调模型→推理→释放，再加载原始模型→推理→释放（8G 单卡方案）。三阶段流程：Qwen决策工具调用→直接 retriever.invoke() 检索法条→RAG增强生成最终回答。支持多轮历史（最近3轮）。|
| `agent_pipeline.py` | **旧推理入口（MCP版）**。通过 asyncio + stdio_client 拉起 server.py 子进程，走 MCP 协议检索。链路完整但复杂，保留用于演示 MCP 全链路。|
| `direct_rag.py` | 轻量检索工具。绕过 LangChain，直接用 chromadb 原生 Python 客户端查询向量库。可单独调用，也可作为 inference_core 的备用检索后端。|
| `test_mcp_server.py` | 独立诊断工具。专门压测 server.py 的 MCP 管道通信，与主流程无关。|

---

### web/（评估平台）

| 文件 | 作用 |
|---|---|
| `app.py` | FastAPI 后端。启动时调用 `load_models()` 加载资源；管理会话（内存字典，保留最近3轮历史）；`POST /chat` 接收问题、调用 `run_inference()`、追加历史并返回双列结果；`POST /session/new` 创建会话；`GET /session/{id}/history` 查询历史。|
| `index.html` | 纯 HTML+JS 前端，无框架依赖。双列布局（左：原始模型，右：微调模型+知识库），右列显示可折叠的推理过程（thought、工具名、检索词、法条片段）。支持多轮对话，底部可展开历史记录面板。|

启动方式：
```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
# 浏览器访问 http://localhost:8000
```

---

### 其他工具文件

| 文件 | 作用 |
|---|---|
| `merge_model.py` | 一次性工具。在 CPU 上加载基座模型 + LoRA 权重（PeftModel），调用 `merge_and_unload()` 合并，保存到 `qwen_merged/`。合并后模型可直接用于推理，无需再加载 LoRA 适配器。|
| `tree.py` | 辅助工具。递归遍历项目目录，用 `├──`/`└──` 绘制 ASCII 树形图，自动排除 `.venv`、`qwen_mcp_lora_output/`、`qwen_merged/` 等大目录。运行 `python tree.py` 即可打印当前目录结构。|
| `untils/compress.py` | 探索性代码。实现 float32 向量 → uint8 的无损压缩与还原：保留最大值索引 + 将剩余值线性映射到 [0,254]。设计目的是压缩 Chroma 向量存储空间，但未接入主流程。|

### version/（历史文档归档）

| 文件 | 作用 |
|---|---|
| `Agent_Distill_ProjectDoc.md` | 初始项目设计文档。定义两大模块（legal_rag / distill）的依赖关系、技术选型理由、完整运行流程，是项目最早期的规划蓝图。|
| `improvement_plan.md` | 改进计划 v1。诊断端到端链路断裂（MCP 进程管理问题），提出用 `inference_core.py` 替代 MCP 方案，并规划 Web 评估平台。|
| `improvement_plan_v2.md` | 改进计划 v2（当前版本）。将工作拆分为三个递进目标：推理层重构 → Web 评估平台 → 多轮对话测试脚本，细化每个目标的输入输出和注意事项。|

---

```bash
# 1. 建立向量库（一次性）
python legal_rag/ingest.py

# 2. 生成训练数据（需 DeepSeek API Key，写在 gen_data.py 第6行）
python distill/gen_data.py

# 3. 微调（需 GPU，约 6G 显存，训练约数小时）
python distill/train.py

# 4. 评估微调效果
python distill/evaluate.py

# 5. 启动 Web 服务
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

---

## 技术栈

| 组件 | 方案 |
|---|---|
| 基座模型 | Qwen2.5-1.5B（本地，离线） |
| 教师模型 | DeepSeek V3（`deepseek-chat`，API） |
| 微调方法 | LoRA（PEFT + TRL SFTTrainer，INT8量化） |
| 向量数据库 | Chroma（本地持久化） |
| Embedding | `shibing624/text2vec-base-chinese`（CPU） |
| PDF 解析 | pymupdf（fitz） |
| Agent 协议 | MCP（FastMCP，保留备用） |
| Web 后端 | FastAPI + uvicorn |
| 前端 | 纯 HTML + JS，无框架 |

---

## 已知问题与待办

### 待处理

**DATA-1：训练数据未做拒绝采样**
`gen_data.py` 生成的 200 条数据直接写入，没有过滤 JSON 格式错误、工具名拼写错误的样本。
修复方式：写入前校验 JSON 可解析 + 工具名在白名单内 + thought 字段不少于20字。

**MODEL-1：训练数据 output 格式非严格 JSON**
当前训练数据 output 字段是中文自然语言混合格式（`"思考：...\n行动：..."`），
导致模型推理时输出格式不稳定，需要正则兜底解析。
根本修复：改 `gen_data.py` 的输出格式为纯 JSON，重新生成数据并重训。

**MODEL-2：尚未做 DPO 偏好对齐**
SFT 之后可用 TRL `DPOTrainer` 再训一轮，用正确工具调用作 chosen、错误调用作 rejected，
预期准确率再提升 5-10 个百分点。

**多轮对话性能未系统测试**
`evaluate_multiturn.py` 尚未编写。需覆盖三类场景：
上下文记忆（追问时引用前轮信息）、追问处理（回答比上轮更细）、话题切换（不带入旧上下文）。

### 已知限制

- **串行推理延迟高**：8G 单卡无法同时驻留两个模型，每次请求需串行加载两个 1.5B 模型，
  单次响应时间约 30-60 秒，不适合高并发，当前仅供评估使用。
- **上下文窗口小**：Qwen2.5-1.5B 上下文 2048 token，法条注入截断至 600 字符，
  多轮历史限制最近 3 轮，长案件追问能力有限。
- **向量库不区分民法/劳动法**：server.py 的两个工具名虽已拆分，但底层共用同一 retriever，
  检索结果取决于 query 内容而非工具名，民法/劳动法的召回精度依赖 query 质量。
