# Phase 5 — Runtime-backed Teacher Trajectory

## Purpose

The teacher must supervise Agent behavior against the real legal tool runtime, rather than inventing tool observations. Each selected tool call is executed through the same service contract used by the Agent Runtime.

## Data flow

```text
user query
  -> DeepSeek teacher
  -> structured tool call
  -> legal tool service / MCP-compatible executor
  -> real observation
  -> teacher next action
  -> ...
  -> final answer
  -> canonical AgentTrajectory JSONL
```

## Stored fields

The collector stores the user query, intent/domain, concise action summaries, tool name and JSON arguments, real observations, and final answer. It deliberately does not store hidden chain-of-thought.

## CLI

Prepare a UTF-8 text file with one legal question per line, then run:

```bash
uv run python -m distill.collect_runtime_trajectories \
  --queries data/teacher_queries.txt \
  --output data/trajectories/teacher.jsonl \
  --max-steps 4
```

`DEEPSEEK_API_KEY`, the DeepSeek endpoint/model, embedding model and data paths are read from the existing settings configuration.

## Quality boundary

The collector validates every completed trajectory against `AgentTrajectory`. Failed or incomplete trajectories are not silently written as training data. The generated JSONL remains a source artifact; SFT conversion is a separate step so data QA can reject samples before training.
