# Agent_Distill 项目文档

> **Phase 1 重构说明**：本项目在 2026-08-25 完成了 Phase 1 模块化重构
> （详见 [`docs/phase1_refactor_notes.md`](docs/phase1_refactor_notes.md)），
> 目录结构从 `legal_rag/ inference/` 等演进为下面这份新结构。项目的长期规划
> 和各阶段进度记录在 [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)。

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
├── configs/                       # 统一配置系统
│   └── settings.py                # 路径 / API Key / 推理参数，优先读环境变量，
│                                   # 不配置时默认值与重构前硬编码值一致
│
├── app/                           # 跨模块共享的横切关注点（不含业务逻辑）
│   ├── logging_config.py          # 统一 logging（替代散落各处的 print）
│   └── exceptions.py              # 统一异常体系
│
├── knowledge/                     # 法律知识库 / RAG 数据层（原 legal_rag/ 的一部分）
│   ├── ingest.py                  # PDF 解析、分块、写入 Chroma（一次性运行）
│   └── direct_rag.py              # 绕过 LangChain 直接查询 Chroma 的轻量检索器（预留/备用）
│
├── mcp_service/                   # MCP 工具服务（原 legal_rag/ 的另一部分；
│                                   # 之所以不叫 mcp/，是因为会和第三方 mcp 包重名，见该目录 __init__.py）
│   ├── server.py                  # MCP 服务端，暴露 search_civil_law / search_labor_law
│   ├── debug_rag.py               # MCP 客户端连通性测试
│   └── test_mcp_server.py         # 独立诊断工具：压测 server.py 的 MCP 管道通信
│
├── distill/                       # Agent 能力蒸馏（数据生成 + 训练 + 合并）
│   ├── tools_config.json          # 工具定义（search_civil_law / search_labor_law）
│   ├── gen_data.py                # 调用 DeepSeek API 生成 CoT + 工具调用训练数据
│   ├── train.py                   # LoRA 微调 Qwen2.5-1.5B
│   ├── merge_model.py             # 一次性工具：CPU 上把 LoRA 合并进基座，导出到 qwen_merged/
│   └── data/
│       └── agent_distill_train.jsonl  # 训练数据（200 条，Alpaca 三段式格式）
│
├── agent/                         # Agent 推理层（原 inference/）
│   ├── inference_core.py          # 主推理入口：直接调 Chroma，串行双模型，供 Web 调用
│   └── agent_pipeline.py          # 旧推理入口：MCP 协议版本，保留备用
│
├── evaluation/                    # 模型评估
│   └── evaluate.py                # 微调前后工具调用准确率对比（JSON 精确匹配 + 正则降级）
│
├── web/                           # Web 评估平台
│   ├── app.py                     # FastAPI 后端：会话管理 + 调用 agent/inference_core
│   └── index.html                 # 前端：双列对比 + 多轮对话 + 推理过程展示
│
├── deployment/                    # 部署配置占位目录（Phase 1 暂无 Docker，见其中 README）
│
├── tests/                         # 结构性单元测试（不依赖 GPU/模型权重/第三方重库）
│   └── test_config_and_core.py
│
├── docs/                          # 文档
│   ├── PROJECT_PLAN.md            # 项目分阶段计划与进度（持续维护）
│   ├── phase1_refactor_notes.md   # Phase 1 重构记录：改了什么、为什么、怎么跑、测试结果
│   ├── architecture/              # Phase 1 之前的代码审计报告 + 目标态规划（历史快照）
│   └── legacy/                    # 更早期的设计文档归档（原 version/ 目录）
│
├── untils/
│   └── compress.py                # float32/uint8 向量压缩与解压工具（探索性代码，未接入主流程）
│
├── tree.py                        # 辅助工具：递归打印项目目录树结构，自动排除模型/虚拟环境目录
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

### configs/（配置系统）

| 文件 | 作用 |
|---|---|
| `settings.py` | 统一 `Settings`（dataclass 单例）。所有模型/数据/向量库路径、DeepSeek API Key、推理参数都从环境变量读取，**读不到环境变量时回退到与重构前完全相同的硬编码默认值**（唯一例外是 `deepseek_api_key`，出于安全原因不再有默认值，见下方"如何运行"）。|

### app/（横切关注点）

| 文件 | 作用 |
|---|---|
| `logging_config.py` | `get_logger(name)`，全项目统一 logging 格式，替代原来各处的 `print`。|
| `exceptions.py` | 统一异常体系（`AgentDistillError` 及其子类），替代原来"裸异常/吞掉异常返回字符串"的错误处理方式。|

### knowledge/（法律知识库数据层）

| 文件 | 作用 |
|---|---|
| `ingest.py` | 用 pymupdf 解析两本法律 PDF，手写分块（500字/50重叠），写入 Chroma 向量库。只需运行一次。|
| `direct_rag.py` | 绕过 LangChain，直接用 chromadb 原生 Python 客户端查询向量库的轻量检索器（预留/备用，当前未被主流程调用）。|

Embedding 模型：`shibing624/text2vec-base-chinese`，运行在 CPU。

### mcp_service/（MCP 工具服务）

| 文件 | 作用 |
|---|---|
| `server.py` | 基于 FastMCP 的 MCP 服务端。加载持久化 Chroma，暴露 `search_civil_law` 和 `search_labor_law` 两个工具，内部共用同一 retriever。|
| `debug_rag.py` | 以 MCP 客户端身份拉起 server.py 子进程，验证握手、工具注册、检索召回全链路是否正常。|
| `test_mcp_server.py` | 独立诊断工具，专门压测 server.py 的 MCP 管道通信，与主流程无关（和 `debug_rag.py` 功能有重叠，两者都保留）。|

---

### distill/（Agent 能力蒸馏）

| 文件 | 作用 |
|---|---|
| `tools_config.json` | 定义两个工具的名称、描述、参数 schema，是数据生成和推理的共享契约。|
| `gen_data.py` | 调用 DeepSeek API（`deepseek-chat`）批量生成训练数据：劳动法场景40%、民法场景40%、负样本20%，循环10批共200条，写入 `distill/data/agent_distill_train.jsonl`。|
| `train.py` | INT8 量化加载 Qwen2.5-1.5B，注入 LoRA（r=8，覆盖全部注意力层和 FFN），用 TRL SFTTrainer 训练3轮，约6G显存，LoRA 权重保存至 `qwen_mcp_lora_output/`。|
| `merge_model.py` | 一次性工具。在 CPU 上加载基座模型 + LoRA 权重（PeftModel），调用 `merge_and_unload()` 合并，保存到 `qwen_merged/`。合并后模型可直接用于推理，无需再加载 LoRA 适配器。|
| `data/agent_distill_train.jsonl` | 训练数据（200 条，Alpaca instruction/input/output 三段式格式）。|

### evaluation/（模型评估）

| 文件 | 作用 |
|---|---|
| `evaluate.py` | 5道测试题（劳动法×2、民法×2、负样本×1），对比原始模型和微调模型的工具调用准确率。主评估为 JSON 字段精确匹配，降级为正则匹配，分别统计两种命中数。|

---

### agent/（推理层）

| 文件 | 作用 |
|---|---|
| `inference_core.py` | **主推理入口**。启动时只加载 tokenizer + Chroma retriever；每次请求串行加载微调模型→推理→释放，再加载原始模型→推理→释放（8G 单卡方案）。三阶段流程：Qwen决策工具调用→直接 retriever.invoke() 检索法条→RAG增强生成最终回答。支持多轮历史（最近3轮）。|
| `agent_pipeline.py` | **旧推理入口（MCP版）**。通过 asyncio + stdio_client 拉起 mcp_service/server.py 子进程，走 MCP 协议检索。链路完整但复杂，保留用于演示 MCP 全链路。|

---

### web/（评估平台）

| 文件 | 作用 |
|---|---|
| `app.py` | FastAPI 后端。启动时调用 `agent.inference_core.load_models()` 加载资源；管理会话（内存字典，保留最近3轮历史）；`POST /chat` 接收问题、调用 `run_inference()`、追加历史并返回双列结果；`POST /session/new` 创建会话；`GET /session/{id}/history` 查询历史。|
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
| `tree.py` | 辅助工具。递归遍历项目目录，用 `├──`/`└──` 绘制 ASCII 树形图，自动排除 `.venv`、`qwen_mcp_lora_output/`、`qwen_merged/` 等大目录。运行 `python tree.py` 即可打印当前目录结构。|
| `untils/compress.py` | 探索性代码。实现 float32 向量 → uint8 的无损压缩与还原：保留最大值索引 + 将剩余值线性映射到 [0,254]。设计目的是压缩 Chroma 向量存储空间，但未接入主流程。|

### docs/legacy/（历史文档归档，原 version/ 目录）

| 文件 | 作用 |
|---|---|
| `Agent_Distill_ProjectDoc.md` | 初始项目设计文档。定义两大模块（legal_rag / distill）的依赖关系、技术选型理由、完整运行流程，是项目最早期的规划蓝图。|
| `improvement_plan.md` | 改进计划 v1。诊断端到端链路断裂（MCP 进程管理问题），提出用 `inference_core.py` 替代 MCP 方案，并规划 Web 评估平台。|
| `improvement_plan_v2.md` | 改进计划 v2。将工作拆分为三个递进目标：推理层重构 → Web 评估平台 → 多轮对话测试脚本，细化每个目标的输入输出和注意事项。|

---

## 如何运行

```bash
# 0. 安装依赖（首次）
uv sync
# 或 pip install -e .

# 1. 建立向量库（一次性）
python knowledge/ingest.py

# 2. 生成训练数据（需 DeepSeek API Key；出于安全考虑，Key 不再硬编码在代码里，
#    必须通过环境变量提供 —— 如果你手上的 Key 是本仓库早期版本里泄露过的那个，
#    请先去 DeepSeek 控制台吊销/轮换，再用新 Key）
export DEEPSEEK_API_KEY="sk-..."
python distill/gen_data.py

# 3. 微调（需 GPU，约 6G 显存，训练约数小时）
#    需要先把 base_model_path / lora_output_dir 指向你本机的实际路径，
#    见下面"环境变量"一节，不配置则使用与重构前相同的默认路径。
python distill/train.py

# 4. （可选）把 LoRA 合并进基座，得到独立可加载的完整模型
python distill/merge_model.py

# 5. 评估微调效果
python evaluation/evaluate.py

# 6. 启动 Web 服务
uvicorn web.app:app --host 0.0.0.0 --port 8000
# 浏览器访问 http://localhost:8000
```

### 环境变量（均可选，不配置时使用与重构前一致的默认值）

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 无默认值，必须配置 | 生成训练数据用的 DeepSeek API Key |
| `AGENT_DISTILL_BASE_MODEL_PATH` | `D:\py\Qwen2.5-1.5B` | 本地 Qwen2.5-1.5B 基座模型路径 |
| `AGENT_DISTILL_LORA_OUTPUT_DIR` | `D:\py\Agent_Distill\qwen_mcp_lora_output` | LoRA 训练输出 / 推理读取目录 |
| `AGENT_DISTILL_MERGED_MODEL_DIR` | `<项目根目录>/qwen_merged` | 合并后模型输出目录 |
| `AGENT_DISTILL_CHROMA_DB_DIR` | `<项目根目录>/knowledge/chroma_db` | 向量库持久化目录 |
| `AGENT_DISTILL_TRAIN_DATA_PATH` | `<项目根目录>/distill/data/agent_distill_train.jsonl` | 训练数据路径 |
| `AGENT_DISTILL_MAX_HISTORY_TURNS` | `3` | 多轮对话保留轮数 |

完整字段定义见 `configs/settings.py`。项目根目录放一份 `.env` 文件也会被自动加载
（需要装 `python-dotenv`，未装则需手动 `export`）。

### 运行测试

```bash
pip install pytest --break-system-packages   # 或 uv add --dev pytest
pytest tests/ -v
```

不依赖 GPU / 模型权重 / torch 等重量级三方库，验证配置系统、异常体系、logging、
模块可导入性。详见 [`docs/phase1_refactor_notes.md`](docs/phase1_refactor_notes.md)。

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

> 以下条目均为 **Phase 1 之前就存在、Phase 1 明确不处理**（Phase 1 范围是模块化目录 +
> 配置/日志/异常系统，不改数据格式、不改训练逻辑、不实现新的 RAG/Router/LangGraph 能力）
> 的问题，留给后续阶段。进度追踪见 [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)。

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
