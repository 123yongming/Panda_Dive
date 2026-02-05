"""CLI runner for LangSmith evaluation of Panda_Dive deep research system.

This module provides a command-line interface for running LangSmith evaluations
on Panda_Dive deep_researcher graph. It supports smoke and full evaluation
modes, with configurable concurrency and timeout settings.

Example usage:
    # Run smoke test on 2 examples
    python tests/run_evaluate.py --smoke

    # Run full evaluation on entire dataset
    python tests/run_evaluate.py --full

    # Run with custom settings
    python tests/run_evaluate.py --full --max-concurrency 3 --timeout-seconds 3600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Add src and tests to path for imports before Panda_Dive import
base_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(base_dir, "..", "src"))
sys.path.insert(0, base_dir)

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langsmith import Client  # noqa: E402
from langsmith.schemas import Example  # noqa: E402

# Load environment variables before Panda_Dive import
load_dotenv()

from Panda_Dive.deepresearcher import deep_researcher  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATASET_NAME = "Deep Research Bench"
DEFAULT_TIMEOUT_SECONDS = 1800  # 30 minutes
DEFAULT_MAX_CONCURRENCY = 1
MAX_CONCURRENCY_CAP = 1
DEFAULT_SMOKE_EXAMPLES = 1
DATASET_ID_PATTERN = re.compile(r"[0-9a-fA-F-]{36}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluation for Panda_Dive deep research system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run smoke test (2 examples)
  python tests/run_evaluate.py --smoke

  # Run full evaluation
  python tests/run_evaluate.py --full

  # Run with custom settings
  python tests/run_evaluate.py --full --max-concurrency 3 --timeout-seconds 3600
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test on small number of examples (default: 2).",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation on entire dataset.",
    )

    # Evaluation configuration
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (overrides smoke default).",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default=None,
        help="Prefix for experiment name (default: auto-generated with timestamp).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=(
            "Dataset name, public URL, or local .jsonl path "
            f"(default: {DEFAULT_DATASET_NAME})."
        ),
    )

    # Execution configuration
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-example timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum concurrent evaluations (default: {DEFAULT_MAX_CONCURRENCY}, max: {MAX_CONCURRENCY_CAP}).",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for evaluation (default: from environment or config).",
    )

    return parser


def get_experiment_name(args: argparse.Namespace) -> str:
    """Generate experiment name based on arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Experiment name string.
    """
    if args.experiment_prefix:
        prefix = args.experiment_prefix
    else:
        prefix = "deep-research-eval"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "smoke" if args.smoke else "full"

    return f"{prefix}-{mode}-{timestamp}"


def normalize_dataset_identifier(dataset_name: str) -> str:
    """Normalize dataset identifier from name or public URL.

    Args:
        dataset_name: Dataset name, UUID, or LangSmith public URL.

    Returns:
        Dataset name or UUID suitable for LangSmith API.
    """
    if not dataset_name:
        return dataset_name

    if dataset_name.startswith("http"):
        parsed = urlparse(dataset_name)
        match = DATASET_ID_PATTERN.search(parsed.path)
        return match.group(0) if match else dataset_name

    match = DATASET_ID_PATTERN.search(dataset_name)
    return match.group(0) if match else dataset_name


def _is_public_dataset_url(dataset_name: str) -> bool:
    if not dataset_name or not dataset_name.startswith("http"):
        return False
    parsed = urlparse(dataset_name)
    return "/public/" in parsed.path and parsed.path.endswith("/d")


def _is_local_jsonl_path(dataset_name: str) -> bool:
    if not dataset_name:
        return False
    path = Path(dataset_name)
    return path.suffix.lower() == ".jsonl" or path.exists()


def _load_local_jsonl_examples(
    file_path: str, max_examples: int | None
) -> list[Example]:
    dataset_id = uuid.uuid5(uuid.NAMESPACE_URL, str(Path(file_path).resolve()))
    examples: list[Example] = []
    with open(file_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not prompt:
                logger.warning("Skipping record without prompt: %s", record)
                continue
            examples.append(
                Example(
                    id=uuid.uuid4(),
                    dataset_id=dataset_id,
                    inputs={
                        "messages": [
                            {"role": "user", "content": prompt},
                        ]
                    },
                    outputs=None,
                    metadata={
                        "id": record.get("id"),
                        "topic": record.get("topic"),
                        "language": record.get("language"),
                    },
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def resolve_dataset_data(
    client: Client,
    dataset_name: str,
    max_examples: int | None,
) -> tuple[Any, bool, str]:
    """Resolve dataset name to dataset object.

    Args:
        client: LangSmith client instance.
        dataset_name: Name or URL of the dataset.
        max_examples: Maximum number of examples to use.

    Returns:
        Tuple of (dataset_object, is_cloned_flag, dataset_name).
    """
    if _is_public_dataset_url(dataset_name):
        dataset = client.clone_public_dataset(dataset_name)
        return dataset.name, True, dataset.name

    if _is_local_jsonl_path(dataset_name):
        examples = _load_local_jsonl_examples(dataset_name, max_examples)
        return examples, False, dataset_name

    dataset_identifier = normalize_dataset_identifier(dataset_name)
    return dataset_identifier, True, dataset_identifier


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        SystemExit: If validation fails.
    """
    # Validate max_concurrency
    if args.max_concurrency > MAX_CONCURRENCY_CAP:
        logger.error(
            f"max-concurrency ({args.max_concurrency}) exceeds cap ({MAX_CONCURRENCY_CAP})"
        )
        sys.exit(1)

    if args.max_concurrency < 1:
        logger.error("max-concurrency must be at least 1")
        sys.exit(1)

    # Validate timeout
    if args.timeout_seconds < 1:
        logger.error("timeout-seconds must be at least 1")
        sys.exit(1)

    # Validate max_examples if provided
    if args.max_examples is not None and args.max_examples < 1:
        logger.error("max-examples must be at least 1")
        sys.exit(1)

    # Check for required environment variables
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.warning("LANGSMITH_API_KEY not set. LangSmith evaluation may fail.")


def create_target_function(args: argparse.Namespace):
    """Create the target function for LangSmith evaluation.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Async function that takes inputs and returns outputs.
    """

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the deep_researcher graph on the input.

        Args:
            inputs: Dictionary containing the input messages.

        Returns:
            Dictionary containing the research output.
        """

        def _extract_from_messages(message_list: list[Any]) -> str | None:
            for msg in message_list:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    msg_type = msg.get("type")
                    content = msg.get("content")
                    if not isinstance(content, str) or not content.strip():
                        data = msg.get("data")
                        if isinstance(data, dict):
                            data_content = data.get("content")
                            if isinstance(data_content, str):
                                content = data_content
                    if role == "user" or msg_type == "human":
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                elif isinstance(msg, HumanMessage):
                    if msg.content and msg.content.strip():
                        return msg.content.strip()
            return None

        def _extract_query(value: Any) -> str | None:
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                return _extract_from_messages(value)
            if isinstance(value, dict):
                nested_messages = value.get("messages")
                if isinstance(nested_messages, list):
                    extracted = _extract_from_messages(nested_messages)
                    if extracted:
                        return extracted
                for key in ("prompt", "question", "query", "input"):
                    nested_value = value.get(key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value.strip()
            return None

        # Extract query from inputs
        query = None
        messages = inputs.get("messages")
        if isinstance(messages, list):
            query = _extract_from_messages(messages)

        if not query:
            for key in ("prompt", "question", "query", "input"):
                query = _extract_query(inputs.get(key))
                if query:
                    break

        if not query:
            query = _extract_query(inputs)

        if not query:
            logger.warning(
                "Could not extract query from inputs: keys=%s", list(inputs.keys())
            )
            return {"output": "Error: Could not extract query from inputs"}

        # Build configuration
        config: dict[str, Any] = {
            "configurable": {
                "search_api": "duckduckgo",
                "allow_clarification": False,
                "max_researcher_iterations": 3,
                "max_react_tool_calls": 6,
                "max_concurrent_research_units": 2,
            }
        }
        if args.model:
            config["configurable"]["model"] = args.model

        try:
            # Run the deep_researcher graph with timeout
            result = await asyncio.wait_for(
                deep_researcher.ainvoke(
                    {"messages": [HumanMessage(content=query)]},
                    config=config,
                ),
                timeout=args.timeout_seconds,
            )

            # Extract the final report and related fields
            final_report = result.get("final_report", "")
            return {
                "output": final_report,
                "final_report": final_report,
                "raw_notes": result.get("raw_notes", ""),
                "research_brief": result.get("research_brief", ""),
            }

        except asyncio.TimeoutError:
            logger.error(f"Evaluation timed out after {args.timeout_seconds}s")
            return {
                "output": f"Error: Evaluation timed out after {args.timeout_seconds}s"
            }
        except Exception as e:
            logger.exception("Error during evaluation")
            return {"output": f"Error: {str(e)}"}

    return target


def get_eval_config() -> list[Any]:
    """Get the evaluation configuration.

    Returns:
        RunEvalConfig with evaluators.
    """
    # Import evaluators from the evaluators module
    from evaluators import (
        completeness_evaluator,
        correctness_evaluator,
        groundedness_evaluator,
        overall_quality_evaluator,
        relevance_evaluator,
        structure_evaluator,
        supervisor_parallelism,
    )

    return [
        overall_quality_evaluator,
        relevance_evaluator,
        structure_evaluator,
        correctness_evaluator,
        groundedness_evaluator,
        completeness_evaluator,
        supervisor_parallelism,
    ]


async def run_evaluation(args: argparse.Namespace) -> None:
    """Run the LangSmith evaluation.

    Args:
        args: Parsed command-line arguments.
    """
    # Initialize LangSmith client
    client = Client()

    # Get experiment name
    experiment_name = get_experiment_name(args)

    # Determine number of examples
    if args.smoke:
        max_examples = args.max_examples or DEFAULT_SMOKE_EXAMPLES
    else:
        max_examples = args.max_examples

    # Log configuration
    logger.info("=" * 60)
    logger.info("Starting LangSmith Evaluation")
    logger.info("=" * 60)
    logger.info(f"Experiment Name: {experiment_name}")
    dataset_data, upload_results, dataset_display = resolve_dataset_data(
        client, args.dataset_name, max_examples
    )
    logger.info(f"Dataset: {dataset_display}")
    logger.info(f"Mode: {'smoke' if args.smoke else 'full'}")
    if max_examples:
        logger.info(f"Max Examples: {max_examples}")
    logger.info(f"Max Concurrency: {args.max_concurrency}")
    logger.info(f"Timeout: {args.timeout_seconds}s")
    if args.model:
        logger.info(f"Model: {args.model}")
    logger.info("=" * 60)

    # Create target function
    target = create_target_function(args)

    try:
        # Get eval config
        evaluators = get_eval_config()

        # Run evaluation
        results = await client.aevaluate(
            target,
            data=dataset_data,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
            max_concurrency=args.max_concurrency,
            upload_results=upload_results,
            metadata={
                "mode": "smoke" if args.smoke else "full",
                "timeout_seconds": args.timeout_seconds,
                "model": args.model,
            },
        )

        logger.info("=" * 60)
        logger.info("Evaluation Complete!")
        logger.info("=" * 60)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Results: {results}")

    except Exception:
        logger.exception("Evaluation failed")
        raise


def main() -> int:
    """Run CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    try:
        # Run evaluation
        asyncio.run(run_evaluation(args))
        return 0
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130
    except Exception:
        logger.exception("Unhandled exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
