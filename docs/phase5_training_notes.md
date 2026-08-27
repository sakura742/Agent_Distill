# Phase 5 — Qwen Agent Distillation Training

## Scope

This stage turns validated runtime-backed Agent trajectories into a reproducible supervised fine-tuning pipeline. The training code is intentionally independent from the LangGraph runtime.

## Pipeline

```text
DeepSeek teacher trajectory
        ↓
trajectory validation
        ↓
chat-format SFT JSONL
        ↓
Qwen3.5-2B base model
        ↓
LoRA / PEFT
        ↓
adapter checkpoint
        ↓
Phase 4 Agent Runtime
```

## Training entry point

```bash
uv run python -m distill.train_lora \
  --model /path/to/Qwen3.5-2B \
  --train-file data/trajectories/train_sft.jsonl \
  --output-dir outputs/qwen3.5-2b-legal-agent \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 3 \
  --gradient-checkpointing
```

For an RTX 4060 8 GB, start with batch size 1, gradient accumulation, gradient checkpointing, and a conservative sequence length. The exact feasible sequence length and precision must be verified on the target machine before a long run.

## Training data contract

Each JSONL row has a `messages` field. A trajectory contributes:

- system instruction;
- user question;
- concise action summaries;
- structured tool calls as JSON;
- real tool observations;
- final answer.

The dataset deliberately avoids requiring the model to reproduce hidden chain-of-thought. The supervision focuses on observable Agent behavior: intent/domain decisions, tool selection, arguments, observations, and final answers.

## Reproducibility

Training parameters are CLI arguments rather than hard-coded paths. The resulting directory is a PEFT adapter and tokenizer artifacts; the base model remains a separately managed dependency.

## Acceptance criteria

Phase 5 training is not considered validated merely because the script starts. Acceptance requires:

1. base model loading smoke test;
2. a tiny overfit/sanity run on a small trajectory subset;
3. adapter checkpoint creation;
4. inference with the adapter;
5. re-attachment to the Phase 4 runtime;
6. later Phase 6 evaluation against the same benchmark used for the base model.
