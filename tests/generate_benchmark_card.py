#!/usr/bin/env python3
"""Generate a benchmark card markdown file from LangSmith project runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from Panda_Dive.benchmark_card import build_markdown_card, summarize_runs

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark card from LangSmith project runs."
    )
    parser.add_argument("--project-name", required=True, help="LangSmith project name.")
    parser.add_argument(
        "--dataset-name",
        default="Deep Research Bench",
        help="Dataset name shown in the benchmark card.",
    )
    parser.add_argument(
        "--model-name",
        default="unspecified",
        help="Model name shown in the benchmark card.",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmark-card-v1.md",
        help="Output markdown path.",
    )
    return parser.parse_args()


def main() -> int:
    """Provide CLI entry point."""
    args = parse_args()

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise RuntimeError("LANGSMITH_API_KEY is required to generate benchmark card.")

    client = Client(api_key=api_key)
    runs = list(
        client.list_runs(
            project_name=args.project_name,
            is_root=True,
        )
    )

    filtered_runs = [
        run
        for run in runs
        if getattr(run, "outputs", None)
        and getattr(run, "outputs", {}).get("final_report") is not None
    ]
    summary = summarize_runs(filtered_runs)
    card = build_markdown_card(
        project_name=args.project_name,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        summary=summary,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card, encoding="utf-8")
    print(f"Benchmark card written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

