"""Deterministic metrics used by the Phase 6 Agent benchmark."""
from __future__ import annotations
from collections import Counter
import re
from typing import Any, Sequence


def accuracy(predictions: Sequence, references: Sequence) -> float:
    if not references: return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)


def _normalize_ref(ref: str) -> str:
    """引用比较只看"法律名+条款序号"，忽略 " | 文件名 p.页码" 这段索引实现
    细节。之前 gold 是完全无关的英文标签（如 "labor_overtime_regulation"），
    从没跟检索器实际产出的引用格式对齐过，导致 recall/mrr/citation_* 恒为0；
    现在 gold 改用和检索结果同源的"法律名 条款号"格式后，如果还按完整字符
    串（含文件名、页码）比较，未来只要 chunk 切分方式或分页变化，原本正确
    的命中又会被误判为 miss。归一化到"法律名+条款号"更贴近"引用的法条是否
    正确"这个业务本意。"""
    return ref.split("|", 1)[0].strip()


def _normalize_refs(refs: Sequence[str]) -> set[str]:
    return {_normalize_ref(r) for r in refs if r}


def recall_at_k(retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]], k: int = 5) -> float:
    if not relevant: return 0.0
    return sum(
        bool(_normalize_refs(gold) & _normalize_refs(got[:k]))
        for got, gold in zip(retrieved, relevant)
    ) / len(relevant)


def mrr(retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]]) -> float:
    if not relevant: return 0.0
    vals=[]
    for got, gold in zip(retrieved, relevant):
        gold_norm = _normalize_refs(gold)
        rank=next((i+1 for i,x in enumerate(got) if _normalize_ref(x) in gold_norm), None)
        vals.append(1/rank if rank else 0.0)
    return sum(vals)/len(vals)


def citation_precision(predicted: Sequence[str], gold: Sequence[str]) -> float:
    p,g=_normalize_refs(predicted),_normalize_refs(gold)
    return len(p & g)/len(p) if p else 0.0


def citation_recall(predicted: Sequence[str], gold: Sequence[str]) -> float:
    p,g=_normalize_refs(predicted),_normalize_refs(gold)
    return len(p & g)/len(g) if g else 0.0


def citation_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    return 1.0 if _normalize_refs(predicted) == _normalize_refs(gold) else 0.0


def argument_accuracy(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    """Per-parameter accuracy, rather than requiring the whole JSON to match."""
    keys=set(gold)
    if not keys: return 1.0 if not predicted else 0.0
    return sum(predicted.get(k) == gold.get(k) for k in keys) / len(keys)


def workflow_success(trace: Sequence[dict[str, Any]], verification: dict[str, Any]) -> float:
    required=["intent_analysis","task_planning","tool_decision","tool_execution","retrieval","generation","verification"]
    nodes=[x.get("node") for x in trace]
    return 1.0 if all(n in nodes for n in required) and verification.get("passed",False) else 0.0


def interruption_error(trace: Sequence[dict[str, Any]], verification: dict[str, Any]) -> float:
    nodes=[x.get("node") for x in trace]
    return 0.0 if workflow_success(trace, verification) else 1.0


def f1(precision: float, recall: float) -> float:
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def _tokenize(text: str) -> list[str]:
    """中文法律问答场景没有空格分词，`str.split()` 会把整段中文当成一个
    token，导致 gold/prediction 几乎不可能重叠、F1 恒为 0（与模型质量无关，
    是评估口径 bug）。这里按 Unicode 汉字逐字切分，连续的英文/数字当作
    一个 token 处理，不依赖 jieba 等分词库。
    """
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)


def token_overlap_f1(prediction: str, reference: str) -> float:
    a = Counter(_tokenize(prediction)); b = Counter(_tokenize(reference))
    common=sum((a & b).values()); p=common/sum(a.values()) if a else 0.0; r=common/sum(b.values()) if b else 0.0
    return f1(p,r)
