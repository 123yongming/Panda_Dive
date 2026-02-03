"""Supervisor parallelism evaluation module."""

import argparse
import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Panda_Dive.deepresearcher import deep_researcher_builder

DEFAULT_DATASET_NAME = "ODR: First Supervisor Parallelism"


def right_parallelism_evaluator(outputs: dict, reference_outputs: dict | None) -> dict:
    """Evaluate if the right parallelism was achieved.

    Args:
        outputs: Output dictionary containing tool call information.
        reference_outputs: Reference outputs for comparison.

    Returns:
        Dictionary containing evaluation score and comment.
    """
    tool_call_count = outputs.get("parallel_tool_calls")
    expected_parallel = None
    if reference_outputs:
        expected_parallel = reference_outputs.get("parallel")
    score = None
    if expected_parallel is not None and tool_call_count is not None:
        score = tool_call_count == expected_parallel
    return {
        "key": "right_parallelism",
        "score": score,
        "comment": f"tool_calls={tool_call_count}, expected={expected_parallel}",
    }


async def target(inputs: dict):
    """Target function for parallelism evaluation.

    Args:
        inputs: Input dictionary containing user messages.

    Returns:
        Dictionary containing parallelism information.
    """
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config,
    )

    parallel_tool_calls = None
    try:
        state = graph.get_state(config, subgraphs=True)
        supervisor_messages = state.values.get("supervisor_messages", [])
        if supervisor_messages:
            parallel_tool_calls = len(supervisor_messages[-1].tool_calls)
    except Exception:
        state = None

    return {
        "parallel_tool_calls": parallel_tool_calls,
        "state_available": state is not None,
    }


async def main():
    """Run supervisor parallelism evaluation.

    Returns:
        Evaluation results.
    """
    load_dotenv()
    os.environ.setdefault("LANGSMITH_PROJECT", "open_deep_research_local")

    parser = argparse.ArgumentParser(
        description="Evaluate supervisor parallelism with a minimal runnable check"
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    client = Client()
    return await client.aevaluate(
        target,
        data=args.dataset_name,
        evaluators=[right_parallelism_evaluator],
        experiment_prefix="Panda_Dive Parallelism",
        max_concurrency=1,
    )


if __name__ == "__main__":
    results = asyncio.run(main())
