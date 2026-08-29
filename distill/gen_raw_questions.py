"""Phase 5: 生成 distill/trajectory.py 需要的原始问题种子（raw_question 文档）。

背景（Phase 5 遗留问题）：trajectory.py 的 CLI 要求传入一个 JSONL 文件，
每行含 input/question/user_query 三者之一作为问题种子，但仓库里从未提交过
这个文件，也没有对应文档说明它从哪来。本脚本补齐这一环。

注意：
- 不要用 data/evaluation/benchmark.jsonl 的问题作为种子——那是 Phase 6 的
  评估集，混进训练轨迹会造成 train/eval 数据污染，Phase 6 的 Raw/LoRA
  对比会失去可信度。
- distill/data/agent_distill_train.jsonl（Phase 1 gen_data.py 产物）的
  "input" 字段是同域、独立生成的问题，可以安全复用；本脚本默认只生成
  全新问题，不依赖它。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from app.exceptions import ConfigurationError, DataGenerationError
from app.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)

if not settings.deepseek_api_key:
    raise ConfigurationError(
        "未设置 DEEPSEEK_API_KEY 环境变量。请先配置：\n"
        '  export DEEPSEEK_API_KEY="sk-..."'
    )

client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

PROMPT = """请模拟用户日常法律纠纷咨询场景，生成 20 条完全不重复的问题。
比例：
1. 劳动法咨询 约 40%（加班费、降薪、裁员赔偿、未签劳动合同等）
2. 民法典咨询 约 40%（租房合同纠纷、财产分割、邻里侵权、借款不还等）
3. 日常闲聊负样本 约 20%（与法律无关的寒暄，如"你好"、AI 话题等）

只返回一个 JSON 数组，元素为字符串问题本身，不要包含解释、不要 markdown 标记。"""

OUTPUT = settings.project_root / "distill" / "data" / "phase5_raw_questions.jsonl"


def _load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["user_query"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def generate_batch() -> list[str]:
    try:
        resp = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.9,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[len("json"):]
            text = text.strip()
        questions = json.loads(text)
        if not isinstance(questions, list):
            raise DataGenerationError(f"期望 JSON 数组，实际收到：{type(questions)}")
        return [q for q in questions if isinstance(q, str) and q.strip()]
    except Exception as exc:
        raise DataGenerationError(f"批次生成失败: {exc}") from exc


def main(num_batches: int = 10) -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    seen = _load_existing(OUTPUT)
    written = 0
    with OUTPUT.open("a", encoding="utf-8") as f:
        for i in range(1, num_batches + 1):
            try:
                batch = generate_batch()
            except DataGenerationError as exc:
                logger.error("第 %d 批次失败: %s", i, exc)
                continue
            new_in_batch = 0
            for q in batch:
                if q in seen:
                    continue
                f.write(json.dumps({"user_query": q}, ensure_ascii=False) + "\n")
                seen.add(q)
                written += 1
                new_in_batch += 1
            logger.info(
                "第 %d 批次完成：新增 %d 条（累计 %d 条）", i, new_in_batch, len(seen)
            )
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=10, help="生成批次数，每批约 20 条")
    args = parser.parse_args()
    count = main(args.batches)
    print(f"写入 {count} 条新问题到 {OUTPUT}")
