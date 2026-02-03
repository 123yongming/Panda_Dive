"""Evaluation runner for Deep Research Bench."""

import argparse
import asyncio
import os
import re
import sys
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Panda_Dive import Configuration
from Panda_Dive.deepresearcher import deep_researcher_builder
from tests.evaluators import (
    eval_completeness,
    eval_correctness,
    eval_groundedness,
    eval_overall_quality,
    eval_relevance,
    eval_structure,
)

DEEP_RESEARCH_BENCH_URL = (
    "https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d"
)
DEFAULT_DATASET_NAME = "deep_research_bench"
SMOKE_OVERRIDES: dict[str, object] = {}


def _parse_public_dataset_id(url: str) -> str | None:
    match = re.search(r"/public/([0-9a-fA-F-]+)/d", url)
    if not match:
        return None
    return match.group(1)


def _resolve_dataset(client: Client, dataset_name: str, dataset_id: str | None):
    if dataset_name:
        try:
            return client.read_dataset(dataset_name=dataset_name)
        except Exception:
            pass
    if dataset_id:
        try:
            return client.read_dataset(dataset_id=dataset_id)
        except Exception:
            pass
    return client.clone_public_dataset(
        DEEP_RESEARCH_BENCH_URL, dataset_name=dataset_name
    )


async def target(inputs: dict):
    """Target function for evaluation.

    Args:
        inputs: Input dictionary containing user messages.

    Returns:
        Dictionary containing evaluation results.
    """
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    if SMOKE_OVERRIDES:
        config["configurable"].update(SMOKE_OVERRIDES)
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config,
    )
    return {
        "final_report": final_state.get("final_report", ""),
        "research_brief": final_state.get("research_brief", ""),
        "raw_notes": final_state.get("raw_notes", []),
        **final_state,
    }


async def main():
    """Run Deep Research Bench evaluation.

    Returns:
        Evaluation results.
    """
    load_dotenv()
    os.environ.setdefault("LANGSMITH_PROJECT", "panda_dive_local")

    parser = argparse.ArgumentParser(description="Run Deep Research Bench evaluation")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--experiment-prefix", default="Panda_Dive Eval")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument(
        "--search-api",
        choices=["tavily", "duckduckgo", "none"],
        default=None,
    )
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.smoke:
        overrides.update(
            {
                "allow_clarification": False,
                "max_researcher_iterations": 1,
                "max_react_tool_calls": 1,
                "max_concurrent_research_units": 1,
                "search_api": "none",
            }
        )
    if args.limit and args.limit > 0:
        overrides.setdefault("max_concurrent_research_units", 1)
    if args.search_api:
        overrides["search_api"] = args.search_api
    global SMOKE_OVERRIDES
    SMOKE_OVERRIDES = overrides

    client = Client()
    dataset_id = args.dataset_id or _parse_public_dataset_id(DEEP_RESEARCH_BENCH_URL)
    dataset = _resolve_dataset(client, args.dataset_name, dataset_id)

    evaluators = [
        eval_overall_quality,
        eval_relevance,
        eval_structure,
        eval_correctness,
        eval_groundedness,
        eval_completeness,
    ]

    config = Configuration.from_runnable_config(None)
    metadata = {
        "allow_clarification": config.allow_clarification,
        "max_concurrent_research_units": config.max_concurrent_research_units,
        "search_api": config.search_api.value,
        "max_researcher_iterations": config.max_researcher_iterations,
        "max_react_tool_calls": config.max_react_tool_calls,
        "summarization_model": config.summarization_model,
        "summarization_model_max_tokens": config.summarization_model_max_tokens,
        "research_model": config.research_model,
        "research_model_max_tokens": config.research_model_max_tokens,
        "compression_model": config.compression_model,
        "compression_model_max_tokens": config.compression_model_max_tokens,
        "final_report_model": config.final_report_model,
        "final_report_model_max_tokens": config.final_report_model_max_tokens,
    }

    if args.limit and args.limit > 0:
        examples = list(client.list_examples(dataset_id=dataset.id, limit=args.limit))
        data = examples
    else:
        data = dataset.name

    max_concurrency = args.max_concurrency
    if max_concurrency is None:
        max_concurrency = 1 if args.smoke else 10

    return await client.aevaluate(
        target,
        data=data,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        max_concurrency=max_concurrency,
        metadata=metadata,
    )


if __name__ == "__main__":
    results = asyncio.run(main())
