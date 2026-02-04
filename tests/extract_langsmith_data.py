#!/usr/bin/env python3
"""CLI exporter to extract LangSmith experiment results to JSONL format.

This script extracts data from a LangSmith project, maps runs to dataset examples,
and outputs a JSONL file with id, prompt, and article fields.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract LangSmith experiment results to JSONL format"
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="LangSmith project name containing the experiment runs",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name for output filename",
    )
    parser.add_argument(
        "--dataset-name",
        default="Deep Research Bench",
        help="Dataset name (default: 'Deep Research Bench')",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/expt_results/",
        help="Output directory (default: tests/expt_results/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file if it exists",
    )
    return parser.parse_args()


def get_langsmith_client() -> Client:
    """Initialize LangSmith client from environment.

    Returns:
        Initialized LangSmith Client.

    Raises:
        SystemExit: If LANGSMITH_API_KEY is not set.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.error("LANGSMITH_API_KEY environment variable is not set")
        sys.exit(1)
    return Client(api_key=api_key)


def extract_data(
    client: Client,
    project_name: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Extract data from LangSmith project runs.

    Args:
        client: LangSmith client instance.
        project_name: Name of the project to extract runs from.
        dataset_name: Name of the dataset (for logging only).

    Returns:
        List of dictionaries with id, prompt, and article fields.

    Raises:
        SystemExit: If project is not found.
    """
    logger.info(f"Extracting data from project: {project_name}")

    try:
        client.read_project(project_name=project_name)
    except Exception as e:
        logger.error(f"Project '{project_name}' not found: {e}")
        sys.exit(1)

    try:
        output_runs = client.list_runs(
            project_name=project_name,
            is_root=True,
        )
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        sys.exit(1)

    runs = []
    for run in output_runs:
        if run.outputs is not None and run.outputs.get("final_report") is not None:
            runs.append(run)

    logger.info(f"Found {len(runs)} runs with final_report output")

    output_jsonl = []
    for run in runs:
        record_id = str(run.id)

        prompt = None
        if run.inputs:
            if "inputs" in run.inputs:
                inputs = run.inputs["inputs"]
                if isinstance(inputs, dict):
                    if "messages" in inputs:
                        messages = inputs["messages"]
                        if messages and len(messages) > 0:
                            prompt = messages[0].get("content")
                    elif "prompt" in inputs:
                        prompt = inputs["prompt"]
                    elif "query" in inputs:
                        prompt = inputs["query"]
            elif "messages" in run.inputs:
                messages = run.inputs["messages"]
                if messages and len(messages) > 0:
                    prompt = (
                        messages[0].get("content")
                        if isinstance(messages[0], dict)
                        else str(messages[0])
                    )

        if not prompt:
            logger.warning(f"Could not extract prompt for run {run.id}, skipping")
            continue

        article = run.outputs.get("final_report") if run.outputs else None
        if not article:
            logger.warning(f"No final_report for run {run.id}, skipping")
            continue

        output_jsonl.append(
            {
                "id": record_id,
                "prompt": prompt,
                "article": article,
            }
        )

    logger.info(f"Successfully extracted {len(output_jsonl)} records")
    return output_jsonl


def write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
    force: bool = False,
) -> None:
    """Write records to JSONL file with atomic write.

    Args:
        records: List of dictionaries to write.
        output_path: Path to output file.
        force: Whether to overwrite existing file.

    Raises:
        SystemExit: If file exists and force is False, or write fails.
    """
    if output_path.exists() and not force:
        logger.error(
            f"Output file '{output_path}' already exists. Use --force to overwrite."
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            delete=False,
            dir=output_path.parent,
        ) as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            temp_path = f.name

        os.replace(temp_path, output_path)
        logger.info(f"Data written to {output_path}")
        logger.info(f"Total records: {len(records)}")

    except Exception as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        logger.error(f"Failed to write output file: {e}")
        sys.exit(1)


def main() -> None:
    """Provide main entry point for the CLI."""
    args = parse_args()

    # Initialize LangSmith client
    client = get_langsmith_client()

    # Extract data from LangSmith
    records = extract_data(
        client=client,
        project_name=args.project_name,
        dataset_name=args.dataset_name,
    )

    if not records:
        logger.warning("No records extracted. Exiting.")
        sys.exit(1)

    # Determine output path
    output_filename = (
        f"{args.dataset_name.replace(' ', '_').lower()}_{args.model_name}.jsonl"
    )
    output_path = Path(args.output_dir) / output_filename

    # Write JSONL file
    write_jsonl(records, output_path, force=args.force)


if __name__ == "__main__":
    main()
