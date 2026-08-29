"""Deterministic metrics used by the Phase 6 Agent benchmark."""
from __future__ import annotations
from collections import Counter
import re
from typing import Any, Sequence


def accuracy(predictions: Sequence, references: Sequence) -> float:
    if not references: return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)


def recall_at_k(retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]], k: int = 5) -> float:
    if not relevant: return 0.0
    return sum(bool(set(gold) & set(got[:k])) for got, gold in zip(retrieved, relevant)) / len(relevant)


def mrr(retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]]) -> float:
    if not relevant: return 0.0
    vals=[]
    for got, gold in zip(retrieved, relevant):
        rank=next((i+1 for i,x in enumerate(got) if x in set(gold)), None)
        vals.append(1/rank if rank else 0.0)
    return sum(vals)/len(vals)


def citation_precision(predicted: Sequence[str], gold: Sequence[str]) -> float:
    p,g=set(predicted),set(gold)
    return len(p & g)/len(p) if p else 0.0


def citation_recall(predicted: Sequence[str], gold: Sequence[str]) -> float:
    p,g=set(predicted),set(gold)
    return len(p & g)/len(g) if g else 0.0


def citation_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    return 1.0 if set(predicted) == set(gold) else 0.0


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
