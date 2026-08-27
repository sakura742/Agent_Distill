# Phase 6 Benchmark / Evaluation

Phase 6 evaluates observable Agent capability rather than hidden reasoning.

## Six benchmark tracks

1. **Routing** — domain/intent routing accuracy.
2. **Retrieval** — Recall@K and MRR against gold document IDs.
3. **Tool Calling** — tool selection and argument accuracy.
4. **Workflow** — expected state transition / task success rate.
5. **Answer** — reference-aligned answer quality; baseline token-overlap F1 is provided, while judge-based evaluation can be added later.
6. **Multi-turn** — conversation-level success rate.

Citation quality is tracked separately with citation precision/recall.

## JSONL

See `benchmark_schema.json`. A minimal routing record is:

```json
{"category":"routing","question":"公司裁员不给赔偿怎么办？","gold":"labor","prediction":"labor"}
```

## Run

```bash
python -m evaluation.run_benchmark evaluation/benchmark.jsonl
```

The evaluator intentionally does not manufacture benchmark numbers. Metrics are emitted only from actual benchmark records.
