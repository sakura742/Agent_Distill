"""Pure-Python metrics used by Phase 6 benchmarks."""
from __future__ import annotations
from collections import Counter
from typing import Sequence


def accuracy(predictions: Sequence, references: Sequence) -> float:
    if not references: return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)


def recall_at_k(retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]], k: int = 5) -> float:
    if not relevant: return 0.0
    vals=[]
    for got, gold in zip(retrieved, relevant):
        g=set(gold); vals.append(1.0 if g and g.intersection(got[:k]) else 0.0)
    return sum(vals)/len(vals)


def mrr(retrieved: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]]) -> float:
    if not relevant: return 0.0
    vals=[]
    for got, gold in zip(retrieved, relevant):
        g=set(gold); rank=next((i+1 for i,x in enumerate(got) if x in g), None)
        vals.append(1.0/rank if rank else 0.0)
    return sum(vals)/len(vals)


def citation_precision(predicted: Sequence[str], gold: Sequence[str]) -> float:
    p=set(predicted); g=set(gold)
    return len(p & g)/len(p) if p else 0.0


def citation_recall(predicted: Sequence[str], gold: Sequence[str]) -> float:
    p=set(predicted); g=set(gold)
    return len(p & g)/len(g) if g else 0.0


def f1(precision: float, recall: float) -> float:
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def token_overlap_f1(prediction: str, reference: str) -> float:
    a=Counter(prediction.split()); b=Counter(reference.split())
    common=sum((a & b).values())
    p=common/sum(a.values()) if a else 0.0
    r=common/sum(b.values()) if b else 0.0
    return f1(p,r)
