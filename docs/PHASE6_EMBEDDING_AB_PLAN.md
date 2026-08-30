# Phase 6 Embedding A/B 实验计划

## 当前证据

同一 retrieval gold set 的 Top-K 实验已经证明：Top-10/Top-20 没有继续改善，且 Civil 的“买卖合同逾期交货”和“邻居漏水”在 Top-20 仍未命中。因而这两个 case 不是单纯的 top-k/ranking 问题。

Rewrite 对“借款”类查询产生了明显收益，说明 query formulation 是有效变量；但 #005/#007 在不同 rewrite/rerank 配置下仍失败，因此需要直接测试 embedding/index 能力。

## 实验原则

不同 embedding 模型不能共用同一 Chroma index。每个 embedding 模型必须使用独立 `--chroma-dir` 并从同一 PDF/chunk pipeline 重新 ingest。

每次实验保持以下变量不变：

- 同一 PDF；
- 同一 article-boundary chunking；
- 同一 domain filter；
- 同一 gold benchmark；
- 同一 top-k；
- 同一 rewrite 开关；
- 同一 reranker 开关。

这样才能把指标差异归因到 embedding 模型。

## 候选模型

### 现有 baseline

`shibing624/text2vec-base-chinese`

### Candidate A

`BAAI/bge-m3`

BGE-M3 是多语言、多粒度 embedding 模型，支持 dense/sparse/multi-vector retrieval，模型页面明确建议 hybrid retrieval + reranking。其 dense embedding 维度为 1024，最大长度 8192。urlBGE-M3 模型页https://huggingface.co/BAAI/bge-m3

### Candidate B

`thenlper/gte-large-zh`

GTE-large-zh 是中文 embedding 模型，1024 维、最大长度 512；其模型卡给出 C-MTEB retrieval 指标并支持文本检索场景。urlGTE-large-zh 模型页https://huggingface.co/thenlper/gte-large-zh

## 实验命令

### Baseline index

```powershell
uv run python -m knowledge.ingest --reset --embedding-model "shibing624/text2vec-base-chinese" --chroma-dir knowledge/chroma_text2vec
```

### BGE-M3 index

```powershell
uv run python -m knowledge.ingest --reset --embedding-model "BAAI/bge-m3" --chroma-dir knowledge/chroma_bge_m3
```

### GTE-large-zh index

```powershell
uv run python -m knowledge.ingest --reset --embedding-model "thenlper/gte-large-zh" --chroma-dir knowledge/chroma_gte_large_zh
```

### Benchmark

把对应 `--embedding-model` 和 `--chroma-dir` 传给 retrieval benchmark：

```powershell
uv run python -m evaluation.retrieval_benchmark data/evaluation/retrieval_benchmark_v2.jsonl --top-k 5 --embedding-model "BAAI/bge-m3" --chroma-dir knowledge/chroma_bge_m3 --output data/evaluation/bge_m3_top5.json
```

再分别增加：

```text
--rewrite
--rerank
```

得到 baseline / rewrite / rerank / rewrite+rereank 的 A/B。

## 判定标准

首先看：

1. Civil #005/#007 是否从 Top-20 miss 变为命中；
2. Recall@5 / Recall@10；
3. MRR；
4. Precision@5；
5. 是否引入 Labor 回归。

只有新 embedding 明显提升 Civil 且没有严重降低总体指标，才考虑替换默认 embedding。

如果新 embedding 仍然让 #005/#007 Top-20 miss，则问题继续向 index metadata/topic labels、PDF article content 和 gold 标注正确性排查，而不是继续无限换模型。

## 重要限制

模型页面的通用 benchmark 结果不能直接证明它在本项目法律语料上一定更好；本项目 gold benchmark 才是最终依据。BGE-M3 官方建议使用 hybrid + rerank，但本项目是否受益仍必须实测。citeturn426599search2turn426599search1

## 当前状态

- Top-K sweep 已完成；
- Rewrite 已证明对借款 query 有帮助；
- 当前 embedding-only baseline 的 Civil #005/#007 仍失败；
- embedding A/B 框架已经加入代码；
- 新 embedding 尚未在用户本地实际跑完，不能提前宣称提升。

## 下一阶段

完成至少：baseline、BGE-M3、GTE-large-zh 三组 index + Top-5 benchmark，再决定默认模型。不要在没有结果前直接修改生产默认 embedding。
