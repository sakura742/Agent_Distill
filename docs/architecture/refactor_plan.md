# Agent_Distill → Legal Agent Platform 重构方案

> 目标：将当前面向单机评估 Demo 的 `Agent_Distill`，逐步演进为面向**企业私有化部署**的
> Legal Agent Platform，覆盖：LangGraph Agent Runtime、多领域法律 RAG、MCP Tool Service、
> Hybrid Router、多专业法律 Agent、Qwen3.5-2B 本地 Agent、Agent Trajectory Distillation、
> Benchmark/Evaluation、多轮对话、FastAPI + Web Dashboard、Docker 部署。
>
> 本文档只做规划，不修改任何现有代码。落地时建议按"目录先行、代码渐进迁移"的方式分阶段实施。
>
> **v1.1 更新说明**：本版本对比并吸收了另一份目标态设计文档《法律 Agent 咨询平台.md》中的可取之处
>（详见 §15），主要补充了：Verification/Re-plan 自纠错循环、结构化法条 Metadata、
> RAG 检索专项指标（Recall@K / MRR）、更丰富的 MCP 工具集、Agent Trajectory 蒸馏数据格式、
> 困难样本挖掘闭环、统一执行记录 Schema、Web Dashboard 细分模块、Base/LoRA/Distilled 三方对比实验设计。

---

## 13. 未来目录结构建议

```
legal-agent-platform/
├── apps/
│   ├── api/                          # 【目标9,10】FastAPI 应用（对外服务入口）
│   │   ├── main.py                   # FastAPI app 工厂
│   │   ├── routers/
│   │   │   ├── chat.py               # /chat, /chat/stream（SSE/WebSocket 多轮对话）
│   │   │   ├── sessions.py           # 会话 CRUD（迁移到 Redis/Postgres）
│   │   │   ├── agents.py             # 查询可用 Agent/领域列表
│   │   │   ├── eval.py               # Benchmark 触发/结果查询 API
│   │   │   └── health.py             # 健康检查（替代 debug_rag.py / test_mcp_server.py）
│   │   ├── schemas/                  # Pydantic 请求/响应模型
│   │   ├── deps.py                   # 依赖注入（Runtime、Router、Session Store）
│   │   └── middleware/               # 鉴权、审计日志、限流
│   │
│   └── web/                          # 【目标10】Web Dashboard 前端
│       ├── src/                      # 组件化前端（迁移 web/index.html 的双列对比理念）
│       └── ...
│
├── agent_platform/                   # 核心 Python 包（可 pip install -e 安装）
│   ├── runtime/                      # 【目标1】LangGraph Agent Runtime
│   │   ├── graph.py                  # 主 Graph：Intent → Router → Agent → Planner → Tool → Retrieval → Generate → Verify →(失败)→ Re-plan
│   │   ├── state.py                  # AgentState（Pydantic）：question/domain/intent/plan/tool_calls/tool_results/
│   │   │                             #   retrieved_documents/citations/answer/verification/error（吸收自对比方案 §6）
│   │   ├── nodes/
│   │   │   ├── intent_node.py        # 【新增】意图/诉求类型识别，独立于领域路由（吸收自对比方案）
│   │   │   ├── router_node.py
│   │   │   ├── planner_node.py       # 【新增】任务规划：拆解多步骤任务（如"先算赔偿再查条款"）
│   │   │   ├── retrieval_node.py
│   │   │   ├── tool_call_node.py
│   │   │   ├── generation_node.py
│   │   │   └── verification_node.py  # 【新增】结果校验；失败则回边 planner_node 触发 Re-plan（自纠错循环）
│   │   ├── tracing.py                # 【新增】统一执行记录（AgentRunRecord），见 §13.1
│   │   └── checkpointer.py           # 会话状态持久化（对接 Redis/Postgres）
│   │
│   ├── router/                       # 【目标4】Hybrid Router
│   │   ├── rule_router.py            # 关键词/正则规则路由（复用 _LABOR_KEYWORDS 思路）
│   │   ├── embedding_router.py       # 语义相似度路由（法域分类）
│   │   ├── llm_router.py             # 小模型/Qwen 意图分类兜底
│   │   └── hybrid_router.py          # 规则优先 → 语义次之 → LLM 兜底的融合策略
│   │
│   ├── agents/                       # 【目标5】多专业法律 Agent
│   │   ├── base.py                   # BaseLegalAgent 抽象接口（统一 prompt/工具集/输出 schema）
│   │   ├── civil_law_agent/          # 民法 Agent（合同、婚姻、财产、侵权）
│   │   ├── labor_law_agent/          # 劳动法 Agent
│   │   ├── criminal_law_agent/       # 预留：刑法 Agent
│   │   └── registry.py               # Agent 注册表，供 Router 动态派发
│   │
│   ├── rag/                          # 【目标2】多领域法律知识库 RAG
│   │   ├── ingestion/
│   │   │   ├── pdf_loader.py         # 替代 legal_rag/ingest.py 的分块逻辑
│   │   │   ├── chunkers/
│   │   │   │   ├── fixed_size.py     # 保留旧策略作为对照基线
│   │   │   │   └── clause_aware.py   # 按条款/章节边界分块（法律文本专用）
│   │   │   ├── metadata_schema.py    # 【新增】结构化 Chunk Metadata：domain/law_name/chapter/article/
│   │   │   │                         #   document_type/source/page/effective_date/chunk_id（吸收自对比方案 §9）
│   │   │   └── pipelines.py          # 可配置多法域批量入库，多格式（PDF/Word/HTML/MD/TXT）Loader 统一接入
│   │   ├── stores/
│   │   │   ├── chroma_store.py       # 每个法域独立 collection
│   │   │   └── base_store.py         # 向量库抽象接口（便于未来切换 Milvus/Qdrant/PGVector）
│   │   ├── embeddings/
│   │   │   └── text2vec_provider.py  # 复用 shibing624/text2vec-base-chinese，抽象为可替换 provider
│   │   └── retrievers/
│   │       ├── domain_router_retriever.py  # 按工具名/法域路由到不同 collection（修复现状问题）
│   │       ├── metadata_filter.py    # 【新增】基于 Metadata 的前置过滤（先按 domain/law_name 缩小范围）
│   │       └── reranker.py           # 【新增】Top-K 后重排序，提升 Recall@K / MRR（吸收自对比方案 §10-11）
│   │
│   ├── tools/                        # 【目标3】MCP Tool Service
│   │   ├── contracts/
│   │   │   └── tools_schema.py       # 唯一工具契约来源（Pydantic 模型，替代 tools_config.json 硬编码）
│   │   ├── mcp_server.py             # 常驻 MCP Server（替代每次新起子进程的 legal_rag/server.py 用法）
│   │   ├── mcp_client.py             # 统一 MCP 客户端封装，供 Runtime 调用
│   │   ├── implementations/          # 【扩展】法律工具集，覆盖对比方案 §13 提出的更完整工具面
│   │   │   ├── search_law.py         # 通用法条检索（civil/labor 等按 domain 参数细分）
│   │   │   ├── search_case.py        # 【新增】案例检索
│   │   │   ├── search_judicial_interpretation.py  # 【新增】司法解释检索
│   │   │   ├── analyze_contract.py   # 【新增】合同条款分析
│   │   │   ├── calculate_compensation.py  # 【新增】赔偿/补偿金计算（结构化工具，非纯检索）
│   │   │   └── find_related_articles.py   # 【新增】关联法条查找
│   │   └── parsers/
│   │       └── tool_call_parser.py   # 统一的 JSON/正则容错解析器（抽取自 evaluate.py / inference_core.py 重复逻辑）
│   │
│   ├── models/                       # 【目标6】本地 Agent 模型层
│   │   ├── providers/
│   │   │   ├── qwen_provider.py      # 支持 Qwen2.5-1.5B（兼容旧版）与 Qwen3.5-2B
│   │   │   └── teacher_provider.py   # DeepSeek/其他教师模型统一封装
│   │   ├── serving/
│   │   │   └── vllm_client.py        # 生产 Serving 接入（替代"每次请求现装现卸"的模式）
│   │   └── registry.py               # 模型版本/路径注册表（配置驱动，非硬编码路径）
│   │
│   ├── distillation/                 # 【目标7】Agent Trajectory Distillation
│   │   ├── data_gen/
│   │   │   ├── scenario_sampler.py   # 复用 gen_data.py 的比例采样思路，改为可配置领域/比例
│   │   │   ├── teacher_runner.py     # 教师模型调用，输出结构化 Trajectory（见 §13.2），非单轮 tool_call
│   │   │   └── quality_pipeline.py   # 【新增】数据质量流水线：JSON Schema 校验 → 工具白名单校验 →
│   │   │                             #   参数校验 → 去重 → 质量过滤（吸收自对比方案 §18，比原 rejection_sampling 更细粒度）
│   │   ├── hard_examples/
│   │   │   └── miner.py              # 【新增】困难样本挖掘：从 Evaluation 的 Error Analysis 中按错误类型
│   │   │                             #   （Routing/Tool Selection/Argument/Retrieval/Workflow/Citation）回捞样本，
│   │   │                             #   重新生成 Teacher Trajectory，形成"评估→挖掘→再训练"闭环（吸收自对比方案 §19）
│   │   ├── datasets/
│   │   │   └── schema.py             # 统一 Trajectory 数据 schema（messages/steps 格式，非 Alpaca 三段式）
│   │   ├── training/
│   │   │   ├── sft_trainer.py        # 复用 train.py 超参与 SFTTrainer 用法，路径全部配置化
│   │   │   ├── dpo_trainer.py        # 【新增】对应 README MODEL-2：DPO 偏好对齐
│   │   │   └── configs/*.yaml        # 训练超参配置文件（替代硬编码）
│   │   └── merge/
│   │       └── merge_lora.py         # 复用 merge_model.py，路径配置化
│   │
│   ├── evaluation/                   # 【目标8】Benchmark / Evaluation
│   │   ├── datasets/
│   │   │   ├── legal_benchmark.jsonl # 从 evaluate.py 硬编码 5 题迁移为独立、可扩展数据集
│   │   │   └── splits.py             # 【新增】train/val/test 严格隔离校验，防止评测数据泄漏（吸收自对比方案 §23）
│   │   ├── metrics/
│   │   │   ├── routing.py            # 【新增】Intent/Domain Accuracy、Top-1/Top-2 路由指标
│   │   │   ├── retrieval.py          # 【新增】Recall@K、MRR、检索延迟（吸收自对比方案 §11）
│   │   │   ├── tool_selection.py     # 工具选择准确率 + 参数准确率 + Format Validity（复用现有双层解析容错逻辑）
│   │   │   ├── workflow.py           # 【新增】Task Success Rate、Step Completion Rate（整任务是否走通，而非仅工具选对）
│   │   │   ├── answer_quality.py     # 回答质量/法条引用率（Citation Accuracy）/幻觉率
	< truncated lines 138-247 >
### 阶段 2：RAG 分域重构（对应目标 2：多领域法律知识库 RAG）
1. 把 `legal_rag/ingest.py` 的定长字符分块升级为条款感知分块（`clause_aware.py`），并保留旧策略作为 A/B 基线对照。
2. 为每个法域（民法/劳动法，未来可扩展）建立独立 Chroma collection，`domain_router_retriever.py` 按工具名精确路由。
3. 抽象 `base_store.py` 向量库接口，为未来切换 Milvus/PGVector 等企业级向量库预留扩展点（私有化部署常见需求）。

**产出**：检索精度不再依赖用户 query 自然语言，而是由工具/Agent 路由决定检索范围。

### 阶段 3：Agent Runtime 与 Hybrid Router（对应目标 1、4、5）
1. 用 LangGraph 重写 `inference_core.py` 的"三阶段"流程为显式 Graph：
   `Intent 节点 → Hybrid Router 节点 → 领域 Agent 节点 → Planner 节点 → Tool 节点 → Retrieval 节点 → Generation 节点 → Verification 节点`，
   状态用 `AgentState` 显式建模（替代当前散落在函数参数里的 `history`/`resources` 隐式状态）。
2. **新增自纠错循环**：`Verification` 节点校验回答是否有法条支撑、参数是否合理；校验失败则回边 `Planner` 节点触发 Re-plan，
   重新决策工具调用，而不是像当前 `inference_core.py` 那样"生成即终止、无二次校验"。
3. 实现 Hybrid Router：规则路由（复用 `_LABOR_KEYWORDS` 关键词表思路）→ 语义路由（embedding 相似度）→ LLM 兜底路由，三级融合，替代当前"JSON 解析失败就退化成单一关键词表"的单薄兜底。**新增不确定性处理**：当 Top-1/Top-2 相似度差距过小（如 0.58 vs 0.56）时不强制路由，而是进入多领域 Agent 或升级到更强模型二次判断。
4. 按法律专业领域拆分出多个 `BaseLegalAgent` 子类（民法 Agent、劳动法 Agent，预留刑法/知识产权等），每个 Agent 有独立 prompt、工具集、输出 schema，Router 负责派发。
5. 废弃 `agent_pipeline.py` 每次调用新起子进程的 MCP 用法，统一改为调用阶段 1 中常驻的 MCP Client。
6. 每次 Graph 执行落一条 `AgentRunRecord`（见 §13.1），为阶段 6 的 Web Dashboard 和阶段 5 的 Benchmark 提供统一数据源。

**产出**：一个可观测、可扩展、状态显式、具备自纠错能力的 LangGraph Runtime，取代当前三条并行且互不复用的推理链路。

### 阶段 4：模型层与蒸馏管线升级（对应目标 6、7）
1. 引入 Qwen3.5-2B 作为新的本地 Agent 基座（与 Qwen2.5-1.5B 并存、可配置切换），通过 `models/registry.py` 做版本管理。
2. 把生产推理迁移到常驻 Serving（vLLM/TGI），彻底解决"每次请求现装现卸模型、30–60 秒延迟"的问题；Web 双模型对比场景可用两个独立 Serving 实例并行，而非串行装卸。
3. 重写 `gen_data.py` → `distillation/data_gen/`：
   - 输出格式改为 §13.2 定义的结构化 **Agent Trajectory**（steps + observation + final_answer），而不是单轮 tool_call。
   - 新增 `quality_pipeline.py`：JSON Schema 校验 → 工具名白名单 → 参数校验 → 去重 → 质量过滤，五级流水线（对应 README DATA-1，比原计划更细粒度）。
   - 训练数据 schema 从 Alpaca 三段式迁移为 messages/trajectory 格式，为后续多轮蒸馏与工具轨迹留出扩展空间。
4. `train.py` → `training/sft_trainer.py`：路径全部配置化，补充 `eval_dataset`/`evaluation_strategy`、实验跟踪（wandb/tensorboard）、按时间戳/版本号管理 checkpoint。
5. 新增 `training/dpo_trainer.py`，SFT 后接一轮 DPO 偏好对齐（对应 README MODEL-2 待办）。
6. 用清洗后的高质量数据重新生成训练集并重训，验证 MODEL-1（格式不稳定）问题是否解决。
7. **建立三方对比实验**：Base Qwen（无微调）→ LoRA（单轮 tool_call 微调）→ Agent Distilled（完整 Trajectory 微调），
   在同一 Benchmark 上分别评测，用于量化"轨迹蒸馏"相对"单轮蒸馏"的增量价值，而不只是"微调 vs 不微调"的二元对比。
8. **接入困难样本闭环**：`hard_examples/miner.py` 从阶段 5 的 Error Analysis 结果中按错误类型（Routing/Tool Selection/
   Argument/Retrieval/Workflow/Citation）回捞样本，重新调用教师模型生成 Trajectory，补入训练集，形成
   "评估 → 错误分类 → 困难样本 → 教师重新生成 → 训练 → 再评估"的持续迭代闭环。

**产出**：数据质量有保障、可持续迭代的蒸馏管线 + 可配置模型层 + 生产级 Serving。

### 阶段 5：Benchmark / Evaluation 体系化（对应目标 8）
1. 把 `evaluate.py` 中硬编码的 5 道题迁移为独立、可扩展的 `legal_benchmark.jsonl`，并严格划分 train/val/test（`splits.py`），
   测试集不得参与训练/困难样本回捞之外的任何训练用途，避免评估数据泄漏。
2. **建议初版规模 500–1000+ 条**，按能力维度分层覆盖（可参考量级：Routing ~200、Tool Selection ~200、
   Tool Arguments ~150、Retrieval ~150、Workflow ~100、Multi-turn ~100、Answer/Citation ~100），
   而不是像当前 5 道题那样只覆盖"工具选对没有"这一个维度。
3. 扩展评估维度，每类能力对应独立指标（避免用一个模糊的"Agent Accuracy"代表全部能力）：
   - **Routing**：Intent Accuracy、Domain Accuracy、Top-1/Top-2 准确率
   - **Retrieval**：Recall@K、MRR、平均/P95 检索延迟
   - **Tool Calling**：工具选择准确率（复用现有双层解析容错逻辑，抽成 `parsers/tool_call_parser.py` 公共组件）、参数准确率、输出格式合规率
   - **Workflow**：Task Success Rate（整个任务是否走通，而非仅"工具选对"）、Step Completion Rate
   - **Answer**：Correctness、法条引用率（Citation Accuracy）、幻觉率
   - **Multi-turn**：上下文保持率、话题切换准确率（补齐 README 提到但缺失的 `evaluate_multiturn.py`）
4. 新增 `error_analysis/classifier.py`：按错误类型（Routing/Tool Selection/Argument/Retrieval/Workflow/Citation）
   自动分类失败案例，结果同时驱动阶段 4 的困难样本挖掘闭环。
5. 提供 CLI/CI 可调用的 `benchmark_runner.py`，支持 **Base / LoRA / Agent Distilled 三模型批量对比**，接入自动化回归
   （每次训练/发布前跑一遍，结果落盘可追溯，同时计算绝对提升与相对提升两种口径）。

**产出**：可持续迭代、分维度、可追溯的 Benchmark 体系，取代当前一次性、单维度的手工评估脚本。

### 阶段 6：多轮对话、Web Dashboard 与 Docker 化（对应目标 9、10、11）
1. `web/app.py` 的内存 `sessions` 字典迁移到 Redis/Postgres，配合 LangGraph 的 `checkpointer.py` 做状态持久化，支持多实例水平扩展。
2. 新增结构化长期槽位记忆 `slot_memory.py`（如 `domain/employment_years/salary/termination_reason`），
   追问"那我大概能拿多少"时无需重新分析整轮历史即可恢复关键上下文。
3. Web 前端从单文件 `index.html` 迁移到组件化前端，保留双列对比思路并扩展为企业 Dashboard，细分为：
   - **Chat**：多轮法律咨询、专业 Agent 路由结果、法条引用展示
   - **Agent Trace**：按 §13.1 的 `AgentRunRecord` 实时渲染 Graph 各节点（Intent→Router→Agent→Planner→Tool→MCP→Retrieval→Generation→Verification）执行状态、耗时与错误
   - **RAG Trace**：展示 Top-K 检索结果、rerank 分数、命中法条
   - **Multi-turn State**：可视化当前会话的结构化槽位记忆，验证 Checkpoint 与状态管理真正生效
   - **Model Compare**：Base / LoRA / Agent Distilled 并排对比意图、工具调用、参数、检索、回答与延迟
   - **Evaluation Dashboard**：展示 Benchmark 各维度指标（见阶段 5）
   - **Error Analysis**：按错误类型统计并支持下钻查看完整失败案例
4. 补充鉴权、审计日志、限流中间件，满足企业私有化部署的合规要求。
5. 为 API 服务、MCP Tool Service、训练任务、模型 Serving 分别编写 `Dockerfile`（模型 Serving 使用 GPU 容器），
   用 `docker-compose.yaml`（本地/测试环境，编排 `api + agent + mcp_server + model + rag + vector-db + redis + evaluation + web`）
   与可选 K8s 清单（生产环境）编排各组件。

**产出**：可一键私有化部署、具备完整可观测性的 Legal Agent Platform。

---

### 迁移原则总结

- **不做大爆炸重写**：每个阶段结束后系统都应保持可运行，旧链路（如 `agent_pipeline.py`）在被新 Runtime 完全替代前可并行保留，标注为 deprecated。
- **配置先行**：阶段 0 的路径/密钥配置化是后续所有阶段（尤其 Docker 化）的前提，应最先完成。
- **契约先行**：工具/数据 Schema 的统一（阶段 1、4）应先于 Runtime 重写（阶段 3），避免在不稳定契约上搭建新架构。
- **每阶段都补测试**：`tests/` 目录应与各阶段代码同步增长，而非留到最后统一补齐。

---

## 15. 与《法律 Agent 咨询平台.md》方案的对比

对方文档（以下简称"对比方案"）与本方案（以下简称"本方案"）目标一致，但出发点和侧重点不同，两者互补而非互斥。

### 15.1 核心差异

| 维度 | 本方案（refactor_plan.md） | 对比方案（法律_Agent_咨询平台.md） |
|---|---|---|
| 出发点 | 基于对 `Agent_Distill` 仓库的**实际代码审计**（见 `current_architecture.md`），从当前 13 个具体文件的真实问题出发 | 直接给出**目标终态架构**的完整设计，不绑定当前仓库的具体文件/bug |
| 性质 | 工程落地的**迁移方案**：明确哪个现有文件要改成什么、先改什么后改什么、每阶段结束系统仍可运行 | 产品/技术**设计蓝图**，更像是终态系统的规格说明书 + 简历/答辩故事线 |
| 对已知问题的处理 | 显式列出并优先处理审计中发现的真实问题：硬编码 API Key、`server.py` 两个工具函数体重复、训练数据 `keyword`/`query` 参数名混用、Alpaca 格式与推理期待格式不一致等 | 未提及这些具体问题（因为不是从代码审计出发），只在 §36 给出通用的"API Key 不要硬编码"原则 |
| Agent 执行流程 | 三阶段线性流程（Router→Agent→Tool→Generation），**初版未设计自纠错循环** | 显式设计 **Verification → 失败 → Re-plan** 闭环，以及独立的 Intent/Planner 节点，流程更完整 |
| RAG 检索 | 只到"分域 collection + clause-aware 分块"，**未设计 Metadata 结构、未设计 Rerank、评估指标较笼统** | 明确的 Chunk Metadata schema（law_name/chapter/article/effective_date）、Rerank 环节、Recall@K / MRR 专项指标，检索质量的可衡量性更强 |
| 蒸馏数据格式 | 训练数据从 Alpaca 三段式升级为"messages/trajectory 格式"，**原方案对 Trajectory 内部结构描述较简略** | 给出具体的 `steps + observation + final_answer` 结构，以及五级数据质量流水线（Schema 校验→工具校验→参数校验→去重→质量过滤），细节更完整 |
| 评估体系 | 评估维度基本对齐 README 待办（工具选择、多轮一致性等），**初版 Benchmark 规模/分布未量化** | 给出量化建议（500–1000+ 条，按维度分层配比）、更细的错误分类体系、显式的 Base/LoRA/Distilled 三方对比实验设计 |
| 观测性/Trace | 各模块自行记录，**未设计统一 Trace Schema** | 提出统一的执行记录 JSON（run_id/tool_calls/retrieval/citations/verification/latency/token_usage），供 Trace/Eval/Dashboard 共用 |
| Web Dashboard | 只到"企业 Dashboard"的粗粒度描述 | 细分为 Chat / Agent Trace / RAG Trace / Multi-turn State / Model Compare / Evaluation Dashboard / Error Analysis 七个具体页面 |
| 落地路径 | 明确的 **6 阶段迁移步骤**，每阶段对应现有文件的具体改动，风险从低到高排序 | 8 个 Phase，更偏"目标里程碑"而非"现有代码迁移步骤"，未逐一对应当前仓库文件 |
| 附加内容 | 无 | 包含简历写作原则（真实/可复现/有对照/有提升/指标对应能力）、最终 Demo 案例脚本——这些是求职/汇报向的内容，不属于架构范畴 |

### 15.2 各自的优势

**本方案的优势**：
- 直接建立在真实代码审计基础上，能精确指出"现在哪里错了、为什么错、怎么改"，可执行性强，不会出现"目标架构很美但不知道从哪一行代码开始动手"的问题。
- 迁移路径考虑了风险排序与阶段间可运行性，适合真实工程团队渐进式重构。
- 明确指出了安全问题（硬编码 API Key）和数据质量问题（参数名混用）等对比方案未触及的具体缺陷。

**对比方案的优势**：
- 在**系统设计的完整度**上更细致，尤其是自纠错循环（Verification/Re-plan）、结构化 RAG Metadata、统一 Trace Schema、Agent Trajectory 数据格式、困难样本挖掘闭环、Benchmark 量化规模与分层设计，这些是本方案初版中偏薄弱或未展开的部分。
- 给出了具体的 Web Dashboard 页面拆分和三方模型对比实验设计，对"如何证明系统效果"这件事想得更透彻。
- 附带的评估指标体系（分能力维度、有绝对/相对提升口径）比本方案初版更严谨，适合作为最终验收和成果汇报的标准。

### 15.3 已吸收进本方案的改进点

本次更新（v1.1）已将以下内容整合进 §13、§14：

1. Runtime 增加 `Intent` / `Planner` / `Verification` 节点及 Re-plan 失败回边循环（§13 目录、§14 阶段 3）。
2. RAG 增加结构化 Chunk Metadata schema、Metadata 前置过滤与 Rerank（§13 目录）。
3. MCP 工具集从 2 个检索工具扩展为 6 类工具，含案例检索、司法解释检索、合同分析、赔偿计算等结构化工具（§13 目录）。
4. 新增统一执行记录 `AgentRunRecord`（§13.1），作为 Trace/Eval/Dashboard 的共用数据源。
5. 蒸馏数据格式升级为 Agent Trajectory（§13.2），数据质量流水线细化为五级校验（§14 阶段 4）。
6. 新增困难样本挖掘闭环，连接 Evaluation 的 Error Analysis 与 Distillation 的数据生成（§14 阶段 4）。
7. Evaluation 增加 Routing / Retrieval / Workflow 专项指标、量化的 Benchmark 规模建议、train/val/test 隔离要求、Base/LoRA/Distilled 三方对比实验（§14 阶段 5）。
8. Multi-turn Conversation 增加结构化长期槽位记忆 `slot_memory.py`（§13 目录、§14 阶段 6）。
9. Web Dashboard 细分为 Chat / Agent Trace / RAG Trace / Multi-turn State / Model Compare / Evaluation Dashboard / Error Analysis 七个模块（§14 阶段 6）。

**未吸收的部分**：对比方案中的简历写作原则、面试 Demo 脚本等内容不属于系统架构范畴，未纳入本文档；如有需要建议单独整理为 `docs/project_story.md`，与架构文档分开维护。

*本方案基于对仓库当前代码的实际审计（见 `current_architecture.md`）制定，并吸收了对比方案中的关键设计改进，未对源代码做任何修改。*
