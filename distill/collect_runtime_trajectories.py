"""CLI for collecting DeepSeek teacher traces against the configured legal tools."""
from __future__ import annotations

import argparse
from pathlib import Path

from configs.settings import settings
from mcp_service.retriever_service import build_default_service

from .runtime_teacher import collect_to_jsonl, load_tool_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect runtime-backed legal Agent trajectories")
    parser.add_argument("--queries", type=Path, required=True, help="UTF-8 text file, one query per line")
    parser.add_argument("--tools", type=Path, default=Path("distill/tools_config.json"))
    parser.add_argument("--output", type=Path, default=Path("data/trajectories/teacher.jsonl"))
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()

    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    queries = [line.strip() for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    tools = load_tool_contract(args.tools)
    service = build_default_service()
    count = collect_to_jsonl(queries, tools, service.search, args.output, max_steps=args.max_steps)
    print(f"generated {count} trajectories -> {args.output}")


if __name__ == "__main__":
    main()
