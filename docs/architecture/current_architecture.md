# Agent_Distill — 当前架构审计报告

> 审计范围：GitHub 仓库 `sakura742/Agent_Distill`（审计时 HEAD）。
> 本报告为**只读审计**，未修改任何源代码。

---

## 1. 当前项目目录结构

```
Agent_Distill/
├── data/
│   ├── labor_law.pdf                # 《劳动合同法》原文
│   └── minfa.pdf                    # 《民法典》原文
│
├── legal_rag/                       # RAG 模块
│   ├── ingest.py                    # PDF → 手写分块 → Chroma 向量库（一次性脚本）
│   ├── server.py                    # FastMCP 服务端，暴露 2 个检索工具
│   ├── debug_rag.py                 # MCP 客户端联通性测试
│   └── chroma_db/                   # 向量库持久化目录（.gitignore 未生成/未提交）
│
├── distill/                         # 蒸馏模块
│   ├── tools_config.json            # 工具 JSON Schema（数据生成与推理共享契约）
│   ├── gen_data.py                  # 调 DeepSeek API 生成 CoT + 工具调用训练数据
│   ├── train.py                     # LoRA 微调 Qwen2.5-1.5B（INT8 量化）
│   └── evaluate.py                  # 微调前后工具调用准确率对比
│
├── inference/                       # 推理层（存在 2 条并行链路）
│   ├── inference_core.py            # 主推理入口：直调 Chroma，串行加载双模型
│   ├── agent_pipeline.py            # 旧推理入口：走 MCP stdio 协议，独立可执行
│   ├── direct_rag.py                # 轻量检索器：绕过 LangChain 直接用 chromadb 客户端（未被其他模块引用）
│   └── test_mcp_server.py           # MCP 管道诊断脚本
│
├── web/                             # Web 评估平台
│   ├── app.py                       # FastAPI 后端（会话管理 + /chat）
│   └── index.html                   # 纯 HTML/JS 前端，双列对比 UI
│
├── untils/
│   └── compress.py                  # 向量压缩探索代码，无引用方，含拼写错误（untils）
│
├── version/                         # 历史设计文档归档
│   ├── Agent_Distill_ProjectDoc.md
│   ├── improvement_plan.md
│   └── improvement_plan_v2.md
│
├── merge_model.py                   # 一次性脚本：LoRA 权重合并进基座
├── tree.py                          # 目录树打印工具
├── agent_distill_train.jsonl        # 训练数据（200 条，gen_data.py 产物）
├── pyproject.toml / uv.lock         # uv 依赖管理
├── .python-version                  # 声明 Python >=3.14（与依赖生态存在潜在冲突，见 §12）
└── README.md
```

未提交但运行时会生成的目录：`legal_rag/chroma_db/`、`qwen_mcp_lora_output/`、`qwen_merged/`。

---

## 2. 每个主要 Python 文件的作用

| 文件 | 角色 | 关键行为 |
|---|---|---|
| `legal_rag/ingest.py` | 数据入库 | pymupdf 解析 PDF → 手写滑窗分块（500 字/50 重叠）→ `Chroma.from_documents` 写盘。**一次性运行**，无增量更新能力。 |
| `legal_rag/server.py` | MCP 工具服务端 | 启动时加载 `HuggingFaceEmbeddings("shibing624/text2vec-base-chinese")` 和 Chroma，注册 `search_civil_law` / `search_labor_law` 两个 `@mcp_server.tool()`。**两个工具函数体完全相同**（共用同一个 retriever，实际未按法域路由，仅工具名不同）。 |
| `legal_rag/debug_rag.py` | 诊断脚本 | 用 `stdio_client` 拉起 `server.py` 子进程，验证握手/工具注册/检索链路。开发期工具，非生产代码。 |
| `distill/tools_config.json` | 工具契约 | 定义两个工具的 name/description/parameters（JSON Schema），是 `gen_data.py` 生成数据和推理侧解析的唯一"契约来源"，但**未被推理代码以编程方式加载**（推理侧 system prompt 里工具说明是硬编码字符串，和该文件内容需要人工保持同步）。 |
| `distill/gen_data.py` | 教师数据生成 | 调 DeepSeek（`deepseek-chat`）按 40/40/20（劳动法/民法/负样本）比例生成 20 条/批 × 10 批 = 200 条数据，写入根目录 `agent_distill_train.jsonl`。输出格式是**自然语言拼接的伪 JSON**（`"思考：...\n行动：..."`），不是结构化 JSON。**API Key 硬编码在源码第 7 行**（安全问题，见 §12）。 |
| `distill/train.py` | LoRA 微调 | INT8 量化加载 Qwen2.5-1.5B，`peft.LoraConfig(r=8)` 覆盖全部 attention+FFN 线性层，`trl.SFTTrainer` 训练 3 epoch，输出 LoRA adapter。**模型/数据/输出路径全部是 Windows 绝对路径硬编码**（`D:\py\...`），无法在 Linux/容器环境直接运行。 |
| `distill/evaluate.py` | 效果评估 | 5 道人工构造的测试题，双层解析（JSON 精确匹配 → 正则降级），依次评测原始模型与微调模型（优先加载 merged 模型，否则加载 LoRA）。路径同样硬编码。 |
| `inference/inference_core.py` | **当前主推理入口** | `load_models()` 只加载 tokenizer + Chroma retriever（不占显存）；`run_inference()` 为每次请求**串行**：加载微调模型→三阶段推理（决策工具调用→直接 `retriever.invoke()` 检索→RAG 生成）→释放显存→加载原始模型→生成→释放。支持最近 3 轮历史拼接。路径硬编码。 |
| `inference/agent_pipeline.py` | 旧推理入口（保留备用） | 单次运行的 demo 脚本（无对外函数接口），通过 `asyncio` + `stdio_client` 每次检索都**新启动一个 `server.py` 子进程**完成 MCP 调用，附带工具名/参数的"暴力清洗"逻辑。未被 `web/app.py` 引用。 |
| `inference/direct_rag.py` | 备用检索器 | 定义 `DirectRetriever` 类，直接用 `chromadb.PersistentClient` 查询，绕开 LangChain。**当前未被任何模块实际导入**，是死代码/预留代码。 |
| `inference/test_mcp_server.py` | MCP 诊断 | 与 `debug_rag.py` 功能高度重叠（拉起 `server.py` 子进程、握手、列工具、发起一次真实调用），差异仅在日志详略。**两份几乎重复的诊断脚本**。 |
| `web/app.py` | Web 后端 | FastAPI 单文件：启动时调用 `inference_core.load_models()`；`sessions: dict[str, list]` 内存会话（无持久化、重启即丢、无并发/多进程支持）；`POST /chat`、`POST /session/new`、`GET /session/{id}/history` 三个路由；直接把 `web/index.html` 用 `FileResponse` 返回，无静态资源分离。 |
| `web/index.html` | 前端 | 无框架纯 HTML+JS，双列对比布局，可展开推理过程（thought/tool/query/law_snippet），支持多轮与历史面板。 |
| `merge_model.py` | 一次性工具 | CPU 上 `PeftModel.merge_and_unload()` 合并 LoRA，导出到 `qwen_merged/`。路径硬编码。 |
| `tree.py` | 辅助工具 | 递归打印目录树，与业务逻辑无关。 |
| `untils/compress.py` | 探索性代码 | float32→uint8 有损压缩/还原，纯脚本（无 `if __name__`保护，import 即执行 + print），**未接入任何主流程**，目录名拼写错误（`untils` 应为 `utils`）。 |

---

## 3. 当前 Agent Distillation Pipeline

```
DeepSeek(deepseek-chat, teacher)
        │  gen_data.py：40% 劳动法 / 40% 民法 / 20% 负样本，10 批 × 20 条
        ▼
agent_distill_train.jsonl（200 条，字段：instruction / input / output）
        │  output = "思考：{CoT}\n行动：调用 {tool_name}，参数为 {json}" 或 "无需调用"
        ▼
train.py：INT8 量化 Qwen2.5-1.5B + LoRA(r=8, all attn+FFN) + TRL SFTTrainer（3 epoch）
        ▼
qwen_mcp_lora_output/（LoRA adapter）
        │  merge_model.py（可选）
        ▼
qwen_merged/（合并后完整权重，可选产物）
        │
        ▼
evaluate.py：5 题工具调用准确率对比（原始 vs 微调）
```

**关键问题**：训练数据的 `output` 不是严格 JSON（自然语言拼接），而 `evaluate.py` / `inference_core.py` 的 system prompt 却要求模型**严格输出 JSON** `{"thought":..., "tool":{"name":..., "arguments":{"query":...}}}`。即训练目标格式与推理期待格式**不一致**，模型只能靠 few-shot 泛化能力"猜"格式，这是 README 中 MODEL-1 问题的代码级根因。此外经抽样统计，200 条训练数据中约 33 条使用参数名 `keyword`、136 条使用 `query`（两种参数名混用），进一步放大了格式不稳定风险，且未被 README 提及。

蒸馏管线中**没有拒绝采样/数据清洗步骤**（README 中的 DATA-1 待办，代码现状确认：`gen_data.py` 直接 `json.loads` 后写盘，无 schema 校验、无工具名白名单校验、无 thought 长度校验）。

---

## 4. 当前模型训练流程

- 基座：本地离线 Qwen2.5-1.5B（Windows 路径 `D:\py\Qwen2.5-1.5B`）。
- 量化：`BitsAndBytesConfig(load_in_8bit=True)`。
- 微调方法：PEFT LoRA，`r=8, alpha=16, dropout=0.1`，`target_modules` 覆盖 q/k/v/o + gate/up/down 全部线性层。
- 训练框架：`trl.SFTTrainer` + `SFTConfig`（batch=1, grad_accum=4, lr=2e-4, epoch=3, bf16, `optim=paged_adamw_8bit`, `max_length=1024`, `packing=False`）。
- 数据格式化：`formatting_prompts_func` 手写 ChatML 模板（`<|im_start|>system/user/assistant`），未使用 tokenizer 自带 `apply_chat_template`。
- 产物：LoRA adapter 保存到 `qwen_mcp_lora_output/`（Windows 硬编码路径），可选再跑 `merge_model.py` 产出完整权重 `qwen_merged/`。
- **无验证集/无 eval_strategy**（`SFTConfig` 未设置 `eval_dataset` / `evaluation_strategy`），训练全过程无法观测过拟合。
- **无实验跟踪**（`report_to="none"`），无 wandb/tensorboard，不可追溯历次训练超参与结果。
- **无 checkpoint 版本管理**：`save_strategy="epoch"` 每轮覆盖式保存在同一目录，无版本号/时间戳，无法回滚对比。
- 全部路径硬编码为 Windows 盘符路径，**在 Linux/容器/CI 环境下无法直接运行**，也无法通过命令行参数或配置文件覆盖。

---

## 5. 当前 Inference 流程

存在**两条并行、未统一的推理链路**：

### 5.1 `inference_core.py`（Web 服务实际使用的链路）

```
load_models()  ── 启动一次：加载 tokenizer + Chroma retriever（CPU，不占显存）

run_inference(user_query, history):
  ├─ _run_tuned():
  │    ① _load_tuned_model()  加载 base+LoRA 到 GPU
  │    ② 阶段一 Prompt → 生成 JSON 决策（tool + query）
  │    ③ _parse_tool_call()：JSON 解析失败则正则兜底；再失败则用关键词表(_LABOR_KEYWORDS)硬路由
  │    ④ retriever.invoke(query) 直接查 Chroma，截断至 600 字符
  │    ⑤ 阶段三 Prompt（含最近 3 轮历史）→ 生成最终回答
  │    ⑥ _release_model()：del + torch.cuda.empty_cache()
  └─ _run_raw():
       ① _load_raw_model() 重新加载一次原始模型（与上面完全独立的一次 GPU 装载）
       ② 单阶段生成（含历史）
       ③ 释放
```

一次请求 = **两次完整的 GPU 模型装载/卸载**（微调模型一次 + 原始模型一次），是 README 明确记录的已知限制（8G 单卡方案，单次响应 30–60 秒，不支持并发）。

### 5.2 `agent_pipeline.py`（旧链路，未接入 Web，仅可独立运行 `__main__`）

```
一次性加载 base+LoRA 到 GPU（常驻，不释放）
  → 阶段一决策
  → mcp_search()：每次检索都用 asyncio.run() 拉起一个新的 server.py 子进程（stdio），
     初始化 MCP 会话、调用工具、关闭子进程
  → 阶段三生成
```

	< truncated lines 157-162 >

---

## 6. 当前 Evaluation 流程

- 唯一的评估脚本是 `distill/evaluate.py`。
- 测试集是**硬编码在代码里的 5 道题**（劳动法×2、民法×2、负样本×1），非独立数据文件，无法脱离代码复用/扩展。
- 评估维度单一：**仅评估"工具调用是否选对"**（tool name 精确匹配），不评估：
  - 检索 query 关键词质量
  - 最终回答内容质量/事实准确性/是否引用具体法条
  - 多轮对话的上下文一致性（README 明确标注 `evaluate_multiturn.py` 尚未编写，代码库中确认不存在该文件）
  - 延迟、吞吐等系统性指标
- 解析容错分两层（JSON 精确 → 正则降级）并分别计数，但**未做温度=0之外的多次采样鲁棒性测试**（`do_sample=False`，仅测一次贪心解码）。
- 无 CI 集成、无自动化回归、无结果落盘（仅 print 到终端）。

---

## 7. 当前数据集格式

### 7.1 训练数据 `agent_distill_train.jsonl`

```json
{
  "instruction": "你是一个熟练对接后端 MCP 法律知识库服务器的智能助手。请分析用户的法律诉求，给出你的思考过程，并在必要时调用正确的 MCP 工具检索法条。",
  "input": "公司上个月无缘无故降了我30%的工资...",
  "output": "思考：...\n行动：调用法律 MCP 服务器工具 search_labor_law，参数为 {\"keyword\": \"...\"}"
}
```

- Alpaca 式三段（instruction/input/output），**非 ShareGPT/messages 多轮格式**，天然不支持多轮训练样本。
- `output` 是自然语言前缀 + 内嵌 JSON 片段的混合体，不是严格结构化输出（详见 §3）。
- 200 条，抽样统计中工具调用参数名存在 `keyword`（33 条）与 `query`（136 条）**两种不一致命名**，而 `tools_config.json` 中定义的唯一合法参数名是 `query`——即约 1/6 的训练数据本身包含与 schema 不符的错误样本，会直接教坏模型的参数命名习惯。
- 无字段标注样本类别（劳动法/民法/负样本），无法做分层评估或分层采样校验。
- 无来源可追溯字段（如生成批次号、生成时间戳），不利于数据版本治理。

### 7.2 知识库源数据

- `data/labor_law.pdf`、`data/minfa.pdf` 两份 PDF 原文，`ingest.py` 手写滑窗分块（500 字/50 重叠，纯字符级切分，**不感知条款/章节边界**，可能在条文中间截断）。
- 向量库为 Chroma 持久化目录，无 schema 版本、无来源法域字段用于按法域过滤检索。

---

## 8. 当前 Tool Calling 实现

- **未使用标准 Function Calling API**（如 OpenAI/vLLM 的 `tools=[...]` 结构化调用），而是通过**自定义 system prompt + 模型输出裸 JSON 文本 + 手写正则/JSON 解析**来模拟 Tool Calling：

```
system prompt 强制要求输出：
{"thought": "...", "tool": {"name": "...", "arguments": {"query": "..."}}}
```

- 工具定义存在两处、且**不共享同一份数据结构**：
  1. `distill/tools_config.json`（JSON 文件，供 `gen_data.py` 拼进 teacher prompt）
  2. `inference_core.py` / `agent_pipeline.py` / `evaluate.py` 中各自**硬编码的字符串** `_STAGE1_SYSTEM` / `stage1_system` / `SYSTEM_PROMPT`
  - 三处工具描述文案彼此独立维护，容易漂移不一致（已发现：`tools_config.json` 支持 `limit` 参数，但推理侧 system prompt 完全没提及 `limit`，模型也从未学会传该参数）。
- 工具执行侧（真正检索）也存在两种实现：
  1. MCP 协议（`legal_rag/server.py` + `mcp.client.stdio`），仅被 `agent_pipeline.py` / `debug_rag.py` / `test_mcp_server.py` 使用
  2. 直接函数调用 `retriever.invoke()`（`inference_core.py`），**绕开了 MCP，是真正在线上跑的路径**
- 两个"工具"（`search_civil_law`/`search_labor_law`）在 `server.py` 里函数体完全一致，**没有做按法域路由的真实差异化检索**（共用同一 Chroma collection/retriever），工具名的语义与实际行为不符。
- 容错链路很长但都是**字符串层面的补丁**：Markdown 围栏剥离 → JSON 解析 → 正则抓取 `search_(?:labor|civil)_law` → 关键词表兜底（`_LABOR_KEYWORDS`），没有基于 JSON Schema 的结构化校验（如 Pydantic/JSON Schema validate），出错定位困难。

---

## 9. 当前代码之间的依赖关系

```
data/*.pdf
   │
   ▼
legal_rag/ingest.py ──写入──▶ legal_rag/chroma_db/
                                     │
        ┌────────────────────────────┼───────────────────────────┐
        ▼                            ▼                            ▼
legal_rag/server.py         inference/inference_core.py   inference/direct_rag.py（孤立，无调用方）
   │  (MCP 工具)                 (直接 Chroma retriever)
   ▼                                 │
legal_rag/debug_rag.py               │
inference/test_mcp_server.py         │
inference/agent_pipeline.py          │
   （三者都拉起 server.py 子进程）    │
                                     ▼
                              web/app.py ──▶ web/index.html
                                     ▲
distill/tools_config.json ──(人工同步，无代码依赖)──▶ 各处硬编码 system prompt

distill/gen_data.py ──写入──▶ agent_distill_train.jsonl ──被读取──▶ distill/train.py
distill/train.py ──产出──▶ qwen_mcp_lora_output/ ──可选合并──▶ merge_model.py ──▶ qwen_merged/
                                     │                                        │
                                     └──────────────▶ distill/evaluate.py ◀───┘
                                                            │
                                     inference/inference_core.py（同样加载 LoRA/merged）
```

关键依赖特征：
- `web/app.py` 是**唯一的生产入口**，只依赖 `inference/inference_core.py`（`sys.path.insert` 手动加根路径，非标准包结构/无 `__init__.py`，没有可安装的包边界）。
- `agent_pipeline.py`、`direct_rag.py`、`test_mcp_server.py`、`debug_rag.py`、`untils/compress.py` 均为**孤立分支**，不在生产路径上，只在各自 `__main__` 内自成闭环。
- `distill/` 与 `inference/` 之间**没有代码级契约**，只靠双方各自硬编码相同的路径字符串（`qwen_mcp_lora_output`、`D:\py\Qwen2.5-1.5B`）和相同的 system prompt 文案人工保持一致，一旦一方改动另一方不会报错，只会静默失配。

---

## 10. 可以直接复用的代码

| 模块/文件 | 复用方式 | 备注 |
|---|---|---|
| `data/*.pdf` + 法域划分思路 | 直接作为初始知识库语料，接入新的 Ingestion Pipeline | 需要重新分块（按条款切分，而非定长字符切分） |
| `legal_rag/server.py` 的 MCP 工具注册范式（`FastMCP` + `@mcp_server.tool()`）| 作为 MCP Tool Service 的起点 | 需要改为常驻服务、去除与检索逻辑的耦合、真正实现法域分库 |
| `distill/tools_config.json` 的 Schema 思路 | 升级为唯一工具契约来源（Single Source of Truth），供 gen_data / 推理 / MCP server 三方共同加载，而非各自硬编码 | 需要补充 `limit` 等字段的实际使用，去除与硬编码 prompt 的重复 |
| `distill/gen_data.py` 的 teacher-generation 思路（比例采样：正样本×2 类 + 负样本）| 蒸馏数据生成管线的骨架可保留 | 必须重写：输出格式改严格 JSON、加入拒绝采样/schema 校验、去除硬编码 API Key、支持多种法律领域可插拔配置 |
| `distill/train.py` 的 LoRA 超参与 SFTTrainer 用法 | 作为训练脚本模板 | 需要改为可配置路径（CLI/YAML）、加 eval/checkpoint 策略、跨平台（去 Windows 路径） |
| `distill/evaluate.py` 的双层解析容错思路（JSON 精确 + 正则降级）| 可以抽成通用的"输出解析器"工具函数，供 Benchmark 模块复用 | 测试集需要从硬编码迁移为独立数据文件，并扩充评估维度 |
| `inference_core.py` 的三阶段（决策→检索→生成）流程设计 | 作为 LangGraph Agent 节点划分的参考（Router 节点 / Retrieval 节点 / Generation 节点）| 串行加载/卸载模型的实现方式必须重构（见 §11） |
| `web/index.html` 的双列对比 + 可折叠推理过程 UI 思路 | 作为 Web Dashboard 的评估页面原型 | 需要迁移到正式前端框架/组件化，并接入真实多 Agent、多模型对比 |
| `merge_model.py` | 模型合并逻辑本身可复用为一个通用 CLI 工具 | 去除路径硬编码即可 |

---

## 11. 必须重构的代码

| 问题代码 | 问题 | 重构方向 |
|---|---|---|
| 全项目路径硬编码（`D:\py\...`）| 跨平台不可用，无法容器化/CI | 统一改为环境变量 / Pydantic Settings / YAML 配置，Docker 化必须先解决这个 |
| `distill/gen_data.py` 第 7 行硬编码 DeepSeek API Key | **严重安全问题**，密钥已提交进 Git 历史 | 立即改用环境变量/密钥管理服务，并建议使用者**吊销并轮换该密钥** |
| `inference_core.py` 的"每次请求串行加载/卸载两个模型" | 延迟 30–60 秒/请求，不支持并发，无法用于生产 Agent Runtime | 迁移到常驻推理服务（vLLM / TGI / 自建 Serving），模型常驻显存，通过 Router 决定是否需要对比双模型 |
| `agent_pipeline.py` 每次工具调用都新起 MCP 子进程 | 性能极差、且与 `inference_core.py` 逻辑重复但检索方式不同 | 统一为**常驻 MCP Server**（同进程内可用 in-process transport，或独立进程 + 长连接），废弃"每次调用起新进程"的模式 |
| 三处重复且不同步的工具 system prompt（`inference_core.py` / `agent_pipeline.py` / `evaluate.py`）| 维护成本高，语义漂移风险 | 统一从 `tools_config.json`（或其后继 Pydantic 模型）动态渲染 prompt，Single Source of Truth |
| `legal_rag/server.py` 中两个工具函数体完全相同 | 工具名与实际行为不符，检索精度依赖 query 而非法域路由 | 拆分为真正的多 collection / 多 index（民法 collection、劳动法 collection 甚至未来更多领域），工具层做真实路由 |
| 训练数据 output 格式为自然语言拼接的伪 JSON，且参数名 `keyword`/`query` 混用 | 训练目标与推理期待格式不一致，模型输出不稳定 | 重新设计训练数据 schema 为严格 JSON（或原生 tool-calling token 格式，视目标模型如 Qwen3.5 是否原生支持 function calling token），并对既有 200 条数据做清洗/重新生成 |
| `distill/gen_data.py` 无拒绝采样/校验 | 脏数据混入训练集 | 生成后立即做 JSON Schema 校验 + 工具名白名单 + 字段长度校验，不合格样本丢弃并重试 |
| `distill/evaluate.py` 测试集硬编码在代码里、维度单一 | 无法扩展、无法作为持续回归基准 | 迁移为独立 Benchmark 数据集（JSON/YAML），扩充多维度指标（工具选择、参数质量、回答质量、多轮一致性、延迟） |
| `web/app.py` 会话状态为进程内内存字典 | 不支持多实例部署、重启丢失历史、无持久化 | 迁移到 Redis/数据库存储会话，支持水平扩展 |
| `inference/direct_rag.py` 死代码、`untils/compress.py` 未接入的探索代码 | 增加认知负担 | 明确归档到实验分支或直接移除，若保留需补充调用方/测试 |
| `debug_rag.py` 与 `test_mcp_server.py` 高度重复 | 维护两份几乎相同逻辑 | 合并为一个诊断/健康检查工具，纳入未来 MCP Tool Service 的 `/health` 端点 |
| `pyproject.toml` 中 `requires-python = ">=3.14"` | 与 `transformers`/`trl`/`peft` 等生态在 3.14 上的兼容性存在较大不确定性（3.14 发布时间晚于多数 ML 库的官方支持声明），存在环境搭建风险 | 建议锁定到已被 ML 生态广泛验证的 3.10/3.11，并配合 Docker 基础镜像固定 |
| 无测试目录（`tests/`）| 全项目没有一个 `pytest`/单元测试 | 重构过程中为 Router、Tool Parser、RAG 检索、API 层补齐单元测试与集成测试 |

---

## 12. 存在的架构问题（汇总）

1. **无统一 Agent Runtime**：三条推理/工具调用链路（`inference_core.py` 直连 Chroma、`agent_pipeline.py` 走 MCP、`direct_rag.py` 备用检索）并存且互不复用，没有一个统一的状态机/图编排层（这正是目标架构要引入 LangGraph 的原因）。
2. **工具契约不是唯一真源（Single Source of Truth）**：`tools_config.json` 与三处硬编码 system prompt 并存，容易漂移。
3. **训练/推理数据格式不一致**：训练数据是自然语言拼接的伪 JSON，推理期待严格 JSON，直接影响蒸馏效果的可信度。
4. **训练数据质量无保障**：无拒绝采样、无 schema 校验，且已发现参数名混用（`keyword` vs `query`）的真实错误样本混入。
5. **推理性能与生产可用性不匹配**：单请求 30–60 秒、无并发能力、模型每次现装现卸，这是一个"评估用 Demo"架构，而非可服务化的 Runtime。
6. **安全问题**：API Key 硬编码并提交至代码仓库。
7. **跨平台/可部署性差**：几乎所有路径硬编码为 Windows 绝对路径，未做 Docker 化，`.python-version` 声明的 Python 版本与 ML 生态存在兼容性风险。
8. **知识库检索粒度粗**：定长字符分块不感知法律条文边界，且两个"法域工具"共用同一 retriever，检索结果精度依赖用户输入的自然语言而非工具路由本身。
9. **无多领域/多 Agent 扩展能力**：当前只有"民法/劳动法"二选一的单层路由，没有 Hybrid Router、没有按专业领域派发到不同 Agent 的机制。
10. **评估体系单薄**：仅 5 道题、仅评估工具选择正确率，没有覆盖回答质量、多轮一致性、系统性能等目标架构要求的 Benchmark 维度。
11. **Web 层是"评估工具"而非"产品化 Dashboard"**：内存会话、无鉴权、无多用户隔离、无审计日志，不满足企业私有化部署的合规与运维要求。
12. **代码组织缺少工程规范**：无 `tests/`、无 `src/` 包结构、无 `__init__.py`、多处通过 `sys.path.insert` 手动拼路径而非标准可安装包，日志用 `print` 而非 `logging`，无类型检查/CI。
13. **死代码与重复代码并存**：`direct_rag.py`、`untils/compress.py` 未被使用；`debug_rag.py` 与 `test_mcp_server.py` 高度重复；增加了阅读与维护成本。

---

*本报告仅描述现状，重构建议见同目录下的 `refactor_plan.md`。*
