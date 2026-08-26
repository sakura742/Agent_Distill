# Phase 3：RAG 分域重构施工记录

## 目标

解决 Phase 1 遗留的两个核心问题：

1. 所有法律文档进入同一个 Chroma collection，工具名与实际检索范围不一致。
2. 使用全局 500/50 字符滑窗，法律条款边界和引用信息容易丢失。

本阶段不实现 LangGraph，不修改 Agent 推理流程。

## 架构变化

```text
PDF
 ↓
PyMuPDF
 ↓
Page
 ↓
Legal Chunker
 ├── law_name
 ├── chapter
 ├── article
 ├── source
 ├── page
 └── chunk_id
 ↓
Embedding
 ↓
┌──────────────┬──────────────┐
↓              ↓
civil_law      labor_law
↓              ↓
Domain Retriever
 ↓
Metadata Filter
 ↓
Vector Candidate Retrieval
 ↓
Optional CrossEncoder Rerank
```

## 关键设计

### 1. 法域配置

`knowledge/domain_config.py` 是当前法律语料配置入口：

- civil → civil_law → minfa.pdf
- labor → labor_law → labor_law.pdf

后续新增 company / contract / ip 时可以继续增加配置，而不是修改 Retriever 核心逻辑。

### 2. 条款感知分块

优先识别 `第 X 条` 和 `第 X 章/节`，一个法律条款作为主要检索单元；只有单条款超过阈值时才进行局部分段。

这样避免原 500/50 滑窗把一个法律条款从中间截断，并为后续 Citation Accuracy 提供 article/page 元数据。

### 3. Metadata Filter

Retriever 默认限制：

```text
domain = 当前 Agent 法域
```

同时支持：

```text
law_name
article
```

进一步过滤。

### 4. Rerank

第一阶段使用向量召回产生候选集（默认候选数为 `top_k * 4`），再可选使用 Sentence Transformers `CrossEncoder` 对 query-document pairs 二次排序。

通过 `AGENT_DISTILL_RERANKER_MODEL` 配置模型；未配置时保持纯向量检索，避免项目首次运行必须下载额外 reranker。

## 运行

准备真实法律 PDF：

```text
data/minfa.pdf
data/labor_law.pdf
```

重新建库：

```bash
uv run python knowledge/ingest.py --reset
```

检索代码示例：

```python
from knowledge.retriever import LegalRetriever

retriever = LegalRetriever("labor")
results = retriever.search("公司违法解除劳动合同如何赔偿", top_k=5, rerank=False)
```

## 评估

新增 `evaluation/retrieval_metrics.py`：

- Recall@K：前 K 个结果中是否命中任一相关 chunk。
- MRR：第一个相关 chunk 的倒数排名。

本阶段只实现指标计算接口，没有伪造 Benchmark 数字。真正的 Recall@K / MRR 必须使用人工标注的法律问题与 relevant chunk IDs 在本地真实知识库上运行。

## 当前状态

代码层实现完成；真实 PDF 入库、MCP stdio 端到端验证和 Retrieval Benchmark 尚未在远程环境执行，因此 Phase 3 暂保持“进行中”。
