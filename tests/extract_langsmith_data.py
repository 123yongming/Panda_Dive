#!/usr/bin/env python3
"""Extract data from LangSmith and save to JSONL file."""

import argparse
import json
import os

from dotenv import load_dotenv
from langsmith import Client

DEFAULT_DATASET_NAME = "deep_research_bench"


def _extract_prompt(run_inputs: dict | None) -> str:
    if not run_inputs:
        return ""
    if "inputs" in run_inputs and isinstance(run_inputs["inputs"], dict):
        payload = run_inputs["inputs"]
    else:
        payload = run_inputs
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            return first.get("content") or ""
        return str(first)
    return ""


def extract_langsmith_data(
    project_name: str, model_name: str, dataset_name: str, api_key: str
) -> str:
    """Extract data from LangSmith project and save to JSONL file.

    Args:
        project_name: Name of the LangSmith project.
        model_name: Name of the model for output filename.
        dataset_name: Name of the dataset.
        api_key: LangSmith API key.

    Returns:
        Path to the output JSONL file.
    """
    client = Client(api_key=api_key)

    # Verify project exists
    client.read_project(project_name=project_name)
    dataset = client.read_dataset(dataset_name=dataset_name)
    examples = list(client.list_examples(dataset_id=dataset.id))
    examples_dict = {example.id: example for example in examples}

    runs = [
        run
        for run in client.list_runs(project_name=project_name, is_root=True)
        if run.outputs is not None and run.outputs.get("final_report") is not None
    ]

    output_jsonl = []
    for run in runs:
        example = examples_dict.get(run.reference_example_id)
        if not example:
            continue
        output_jsonl.append(
            {
                "id": example.metadata.get("id"),
                "prompt": _extract_prompt(run.inputs),
                "article": run.outputs["final_report"],
            }
        )

    output_file_path = f"tests/expt_results/{dataset_name}_{model_name}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as handle:
        for item in output_jsonl:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_file_path


def main() -> None:
    """Extract LangSmith data from project and save to JSONL file.

    Parses command line arguments and extracts data from LangSmith.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract data from LangSmith project")
    parser.add_argument("--project-name", required=True, help="LangSmith project name")
    parser.add_argument(
        "--model-name",
        help="Model name for output filename (defaults to FINAL_REPORT_MODEL)",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name for output filename",
    )
    parser.add_argument(
        "--api-key",
        help="LangSmith API key (defaults to LANGSMITH_API_KEY env var)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError(
            "API key must be provided via --api-key or LANGSMITH_API_KEY environment variable"
        )

    model_name = args.model_name or os.getenv("FINAL_REPORT_MODEL") or "model"

    extract_langsmith_data(
        project_name=args.project_name,
        model_name=model_name,
        dataset_name=args.dataset_name,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
