"""Run trace generation and Foundry eval in one command.

This script intentionally exposes the same CLI args as `eval/generate_traces.py`
for generation behavior, then runs eval using `eval/foundry/config.yaml`.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from generate_traces import generate


load_dotenv()


def run_generate_and_eval(
    gold_path: Path,
    output_path: Path,
    profile_filter: str | None,
    inter_query_delay: float,
    model_override: str | None,
    temperature: float | None,
    top_p: float | None,
) -> None:
    asyncio.run(
        generate(
            gold_path=gold_path,
            output_path=output_path,
            profile_filter=profile_filter,
            inter_query_delay=inter_query_delay,
            model_override=model_override,
            temperature=temperature,
            top_p=top_p,
        )
    )

    subprocess.run(
        [
            sys.executable,
            "eval/foundry/run_eval.py",
            "--config",
            "eval/foundry/config.yaml",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate traces and run Foundry eval in one command.",
    )
    parser.add_argument(
        "--gold",
        default="eval/datasets/sql_agent_gold_starter.jsonl",
        help="Path to gold dataset JSONL.",
    )
    parser.add_argument(
        "--out",
        default="eval/traces/agent_runs.jsonl",
        help="Output traces JSONL path.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Only run queries for this chat profile (exact name match).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between queries (rate-limit courtesy).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override AZURE_OPENAI_MODEL deployment name (e.g. gpt-4.1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0.0-2.0). Omit to use model default.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (0.0-1.0). Omit to use model default.",
    )
    args = parser.parse_args()

    run_generate_and_eval(
        gold_path=Path(args.gold),
        output_path=Path(args.out),
        profile_filter=args.profile,
        inter_query_delay=args.delay,
        model_override=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
