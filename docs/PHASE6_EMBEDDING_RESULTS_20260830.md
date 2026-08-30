# Phase 6 Embedding A/B 实验结果与交接

## 实验结论

当前本地 A/B 结果由用户在同一 retrieval benchmark 上得到：

- `thenlper/gte-large-zh` 明显优于原 `shibing624/text2vec-base-chinese`，整体 MRR 提升明显，并成功召回买卖合同逾期交货的 gold 第577条。
- `BAAI/bge-m3` 在本项目 benchmark 上没有优于 baseline，部分 labor case 反而 miss，因此不能仅凭通用 benchmark 结论选它。
- 邻居漏水 `#007` 在当前几种 embedding 配置下仍未命中 gold 第1165、1184条；它是当前最顽固的 retrieval hard case。
- 因此后续默认 embedding 基线切换为 `thenlper/gte-large-zh`，但模型名称仍可通过 `AGENT_DISTILL_EMBEDDING_MODEL` 覆盖。

GTE-large-zh 官方模型卡显示其面向中文文本、1024 维、最大输入长度 512 token；其通用 CMTEB retrieval 分数为 72.49。该信息只用于解释模型定位，项目最终选择依据仍然是本地法律 retrieval benchmark。 citeturn594230search0

## 代码变更

### 配置

`configs/settings.py` 默认 embedding 改为：

```text
thenlper/gte-large-zh
```

同时保留环境变量覆盖。

### 索引一致性

`knowledge/ingest.py` 在 Chroma collection metadata 写入：

```text
embedding_model
embedding_normalized
hnsw:space=cosine
```

这样不同 embedding 模型不能在不知情的情况下共用同一个索引目录。

## 下一阶段

### P0：Reranker

优先验证 `BAAI/bge-reranker-v2-m3`。官方模型卡说明它以 query + passage 直接输出 relevance score，属于独立于 embedding 的二阶段 reranker。citeturn594230search2

实验必须保持：

```text
同一 GTE collection
同一 gold
同一 candidate_k
同一 query rewrite
```

只改变 reranker 开关。

### P0：重新生成 GTE 索引后的 baseline

切换到 GTE 后必须重新 `ingest --reset`，再跑 Top-5/10/20。旧 index 不可继续使用。

### P1：#007 hard case

若 GTE + reranker 仍无法召回第1165/1184条，不再继续盲目换 embedding；转向：

- article/topic/scenario metadata
- legal query expansion
- hard-negative 分析
- 必要时增加针对民法典的训练/索引语义标签

### P1：#006 regression

继续监控借款问题，避免为 #007 优化时回退已经改善的 case。

## 禁止事项

- 不因单个 hard case 就把 GTE 改回其它模型。
- 不把通用 CMTEB 分数当成法律领域证明。
- 不在没有 A/B 的情况下把 hybrid/rerank 直接宣称有效。
- 不在 Retrieval 达标前重新训练 LoRA。

## 当前状态

`fix/phase6`：GTE-large-zh 为默认 retrieval baseline；Decision/Answer LoRA 尚未重新训练。
