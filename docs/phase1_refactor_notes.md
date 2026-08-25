# Phase 1 重构记录

> 本文档是 Phase 1（模块化目录结构 + 配置外部化 + 统一日志/异常体系）的权威记录。
> 对应 PR：`phase1-refactor`（初次实现）+ `phase1-fixes`（收尾修复，本文档随后者一并提交）。
> 长期规划与各阶段进度见 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)；重构前的代码审计与目标态
> 规划见 [`architecture/current_architecture.md`](architecture/current_architecture.md) /
> [`architecture/refactor_plan.md`](architecture/refactor_plan.md)。

## Phase 1 的范围（明确不做什么）

按任务要求，Phase 1 **只做**："移动文件 + 建立模块接口"式的重构，**不做**：

- 不删除任何现有功能
- 不修改 `distill/train.py` 的训练核心逻辑（量化配置、LoRA 超参、SFTConfig/SFTTrainer 调用）
- 不改变 `agent_distill_train.jsonl` 的数据集格式（Alpaca instruction/input/output 三段式）
- 不实现 LangGraph / 真正的 MCP 分域路由 / 新的 RAG 能力（这些是 `refactor_plan.md` 里更晚的阶段）
- 不引入 Docker / 微服务拆分（`deployment/` 目前只是占位目录）

## 一、修改了哪些文件

### 1.1 新增的目录骨架

```
app/ agent/ knowledge/ mcp_service/ distill/ evaluation/ web/ deployment/ tests/ configs/ docs/
```

十个模块目录，每个都有 `__init__.py`（内容不是空文件，写了该模块的职责说明）。

唯一偏离任务给定命名清单的地方：`mcp/` 改名为 `mcp_service/`。原因：项目依赖第三方
`mcp` PyPI 包（`from mcp.server.fastmcp import FastMCP` 等），如果本地目录也叫 `mcp/`
并带 `__init__.py`，Python 在项目根目录位于 `sys.path` 时会优先解析到本地目录，
**彻底屏蔽掉真正的第三方 mcp 包**，导致所有 MCP 相关功能在导入阶段直接报错——这违反了
"不删除现有功能""保持项目可运行"的硬性要求，所以改名规避。详见 `mcp_service/__init__.py`。

### 1.2 文件迁移对照表

| 原路径 | 新路径 | 改动内容 |
|---|---|---|
| `legal_rag/ingest.py` | `knowledge/ingest.py` | 路径改用 `configs.settings`；分块算法未改 |
| `legal_rag/server.py` | `mcp_service/server.py` | 向量库路径改用 `configs.settings`；两个工具函数体仍相同（真正分域路由是后续 RAG 阶段的工作，Phase 1 不改此行为）|
| `legal_rag/debug_rag.py` | `mcp_service/debug_rag.py` | 顺手把硬编码相对路径 `"server.py"` 改成基于 `__file__` 的绝对路径（原来隐含假设"CWD 必须是 legal_rag/"，现在从任意目录都能跑）|
| `inference/inference_core.py` | `agent/inference_core.py` | 路径/常量改读 `configs.settings`；`print` 改 `logger`；裸 `FileNotFoundError` 改 `KnowledgeBaseError`；**三阶段推理算法、串行加载/卸载策略、多轮历史拼接逻辑全部未变** |
| `inference/agent_pipeline.py` | `agent/agent_pipeline.py` | 同上；`SERVER_PATH` 指向新的 `mcp_service/server.py` |
| `inference/direct_rag.py` | `knowledge/direct_rag.py` | 路径改用 `configs.settings`；仍然是"未被任何模块调用"的预留代码，原样保留 |
| `inference/test_mcp_server.py` | `mcp_service/test_mcp_server.py` | 路径改用 `configs.settings`；与 `debug_rag.py` 的功能重叠问题保留原状（按"不删除现有功能"原则，去重留给后续阶段）|
| `distill/evaluate.py` | `evaluation/evaluate.py` | 路径改用 `configs.settings`；**测试集内容、双层解析容错逻辑、判定标准完全未变** |
| 根目录 `merge_model.py` | `distill/merge_model.py` | 路径改用 `configs.settings`；合并逻辑（`merge_and_unload()`）未变 |
| 根目录 `agent_distill_train.jsonl` | `distill/data/agent_distill_train.jsonl` | 纯文件搬家，200 条数据内容、Alpaca 三段式格式**逐字节未变**（已用脚本核对行数与首条记录）|
| `version/` | `docs/legacy/` | 三份历史文档整体搬家，内容未改 |

`distill/gen_data.py`、`distill/train.py`、`distill/tools_config.json`、`web/app.py`、
`web/index.html`、`tree.py`、`untils/compress.py` 保持原位置，只做 import/路径修复（见下）。

### 1.3 新增文件

| 文件 | 作用 |
|---|---|
| `configs/settings.py` | 统一配置：所有路径 + DeepSeek API Key + 推理参数，读环境变量，读不到时回退到与重构前完全相同的默认值（`deepseek_api_key` 除外，见 1.4） |
| `app/logging_config.py` | `get_logger(name)`，幂等的全局 logging 配置 |
| `app/exceptions.py` | `AgentDistillError` 及 6 个子类（`ConfigurationError` / `KnowledgeBaseError` / `ModelLoadError` / `ToolCallParseError` / `DataGenerationError` / `MCPConnectionError`）|
| `tests/test_config_and_core.py` | 结构性测试：配置加载、异常体系、logging、目录/模块存在性、业务模块可导入性 |
| `docs/PROJECT_PLAN.md` | 项目分阶段计划与进度跟踪（本次新增，持续维护）|
| `docs/phase1_refactor_notes.md` | 本文档 |

### 1.4 安全修复

`distill/gen_data.py` 第 7 行原来硬编码着 DeepSeek API Key（已提交进 Git 历史）。
修复：删除硬编码值，改为从环境变量 `DEEPSEEK_API_KEY` 读取；读不到时抛出
`ConfigurationError`，报错信息里明确提示"这个 Key 已经泄露过，用之前请先去吊销/轮换"。

**⚠️ 如果你现在用的还是这个仓库早期版本里出现过的那个 Key，请立即去 DeepSeek 控制台
吊销并换一个新的。**

### 1.5 修复历史遗留的相对路径 bug（顺手修，不算"新功能"）

- `legal_rag/debug_rag.py`（现 `mcp_service/debug_rag.py`）原来用相对路径 `args=["server.py"]`
  拉起子进程，隐含"当前工作目录必须是 legal_rag/"这个假设，只要不在该目录下运行就会失败。
  改成基于 `configs.settings.mcp_server_path`（内部用 `__file__` 推导绝对路径），
  现在从项目任意目录运行都能找到。
- `distill/gen_data.py` 原来用裸相对路径 `open("tools_config.json")` 读取工具 schema，
  同样隐含"必须在 distill/ 目录下运行"的假设。改成 `configs.settings.tools_config_path`。

### 1.6 收尾修复（本次新增，即 `phase1-fixes` 分支）

在实际跑 `pytest tests/` 验证时发现并修复了以下 3 个问题：

1. **`distill/train.py` 破坏 pytest 测试会话（已修复）**：原来的"stdout 健壮性修复"
   （把 `sys.stdout` 重新包装为 UTF-8）写在模块顶层，import 该模块时就会执行；
   而 pytest 的 capture 对象恰好也带 `.buffer` 属性，导致这段代码在测试环境下会把
   pytest 内部的 capture buffer "偷梁换柱"，等 pytest 会话收尾时抛出
   `ValueError: I/O operation on closed file`，并连带把同一进程里其他 3 个本该正常
   SKIP 的测试也弄挂（实测：`1 failed, 26 passed, 8 skipped, 7 errors`）。
   修复：把这段代码挪到 `if __name__ == "__main__":` 内部，import 不再有副作用，
   真正把 `train.py` 当脚本跑时行为不变。修复后重跑：`26 passed, 12 skipped`，
   零失败零报错。
2. **根目录 `README.md` 未同步更新（已修复）**：还在引用 `legal_rag/`、`inference/`、
   根目录 `merge_model.py`、`distill/evaluate.py` 等已经不存在的路径，"如何运行"一节
   照着做会直接报"文件不存在"。已按新目录结构全部更新，并补充了环境变量表和测试运行说明。
3. **`docs/phase1_refactor_notes.md` 缺失（已修复）**：`docs/README.md` 和测试文件的
   文件头注释都引用了这份文档，但仓库里没有这个文件。本文档即为补上的内容。

此外顺手给 `pyproject.toml` 补充了 `python-dotenv`（主依赖，支撑 `.env` 自动加载）和
`pytest`（`[dependency-groups] dev`，测试专用，不随主依赖一起装）。

## 二、为什么这样改

- **配置外部化优先于其他一切改动**：8+ 个文件里散落的 Windows 绝对路径硬编码是
  "不可移植、不可容器化"的根源（`current_architecture.md` §11/§12 已指出），
  Docker 化等后续阶段都要先站在一个"路径可配置"的地基上。
- **默认值必须与重构前逐字节一致**：这样即使不配置任何环境变量，项目在原开发者的机器上
  行为也和重构前完全相同——"保持项目可运行"不是靠文档承诺，而是靠默认值可验证地相等
  （见 `tests/test_config_and_core.py::test_settings_loads_with_defaults`）。
- **训练核心逻辑一行不改**：LoRA 超参、量化配置、`formatting_prompts_func` 模板都属于
  "会影响模型效果"的决策，Phase 1 是纯工程重构，任何可能改变模型输出的改动都推迟到
  有专门评估手段的后续阶段（`refactor_plan.md` 阶段 4）再做，避免"重构"和"调参"两件事
  混在一起，出问题时无法定位是哪一步导致的。
- **`evaluate.py` 迁到 `evaluation/` 而不是留在 `distill/`**：它的职能是"评估"而不是
  "生成数据+训练"，任务要求的目录清单里两者本来就是分开的模块；迁移后两者通过
  `configs.settings` 共享路径配置，不再各自硬编码。

## 三、如何运行原来的功能

见根目录 `README.md` 的["如何运行"](../README.md#如何运行)一节（本次一并更新）。
简要版：

```bash
uv sync                                    # 或 pip install -e .
python knowledge/ingest.py                 # 建向量库（原 legal_rag/ingest.py）
export DEEPSEEK_API_KEY="sk-..."
python distill/gen_data.py                 # 生成训练数据（位置未变）
python distill/train.py                    # LoRA 微调（位置未变，核心逻辑未改）
python distill/merge_model.py              # 可选：合并 LoRA（原根目录 merge_model.py）
python evaluation/evaluate.py              # 评估（原 distill/evaluate.py）
uvicorn web.app:app --host 0.0.0.0 --port 8000   # Web 服务（位置未变）
```

所有命令行为与重构前一致，唯一区别是：① 路径改成从环境变量读取（不配置则用原来的
硬编码默认值）；② DeepSeek Key 必须显式配置，不能再靠代码里写死的值蒙混过关。

## 四、测试结果

沙盒环境没有安装 `torch` / `transformers` / `langchain` / `chromadb` / `mcp` / `openai` /
`fastapi` 等重量级三方库，也没有 GPU 和真实模型权重，因此无法在这里真正跑通训练/推理/
检索的端到端流程——这与重构前的状况相同（原代码同样要求这些依赖 + GPU + 本地模型文件）。
能验证、也已经验证的是：**Phase 1 引入的新代码（配置系统、异常体系、logging）本身正确，
且所有业务模块经过迁移后 import 路径没有被改错**。

```
$ pytest tests/ -v
...
26 passed, 12 skipped in 0.11s
```

- **26 passed**：配置默认值与重构前一致、环境变量覆盖生效、`DEEPSEEK_API_KEY` 无硬编码
  默认值、异常体系正确、logging 幂等、10 个模块目录 + 9 个 `__init__.py` 全部存在。
- **12 skipped（预期内）**：`agent.*` / `knowledge.*` / `mcp_service.*` / `distill.*` /
  `evaluation.evaluate` / `web.app` 共 12 个业务模块，因为沙盒里没装 torch/transformers/
  chromadb/mcp/openai/fastapi 而被跳过——**跳过原因是"缺三方库"，不是"import 路径写错"**，
  已经用脱离 pytest 的独立 `python3 -c "import xxx"` 逐个手动核对过，全部只报
  `ModuleNotFoundError`，没有 `AttributeError`/`ImportError: cannot import name` 之类
  说明我们改错了路径或变量名的错误。
- **0 failed / 0 error**：修复 §1.6 的 stdout 副作用问题之前是 `1 failed + 7 error`
  （见 §1.6 的复现记录），修复后清零。

### 未在本沙盒验证、需要使用者在有 GPU + 模型权重的机器上自行确认的部分

- `python distill/train.py` 能否真正跑完一轮 LoRA 训练并保存权重
- `python evaluation/evaluate.py` 的 5 道题准确率数字是否和重构前一致
- `uvicorn web.app:app` 启动后 `/chat` 接口能否正常返回双列结果

这些需要真实的 Qwen2.5-1.5B 权重、GPU、已建好的 Chroma 向量库，不在 Phase 1 的验证范围内
（Phase 1 的验证目标是"重构有没有引入 import/路径层面的 bug"，不是"重新验证模型效果"）。
