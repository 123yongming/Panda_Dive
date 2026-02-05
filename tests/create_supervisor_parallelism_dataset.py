"""Create LangSmith dataset for supervisor parallelism evaluation.

Usage:
    python tests/create_supervisor_parallelism_dataset.py \
        --dataset-name "Panda_Dive: Supervisor Parallelism" \
        --source tests/prompt/supervisor_parallelism.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Load environment variables from .env file before importing langsmith
try:
    from dotenv import load_dotenv

    # Try to load from .env file in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()  # fallback to default behavior
except ImportError:
    pass  # dotenv is optional

from langsmith import Client
from langsmith.schemas import Example

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "prompt", "topic", "language", "reference_outputs")
ALLOWED_LANGUAGES = {"en", "zh"}


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a file."""
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record at line {line_num} must be a JSON object")
            records.append(record)
    return records


def validate_record(record: dict[str, Any], index: int) -> bool:
    """Validate a dataset record has required fields."""
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        logger.error("Record %s missing fields: %s", index, ", ".join(missing))
        return False

    record_id = record.get("id")
    if not isinstance(record_id, int) or record_id < 1:
        logger.error("Record %s: 'id' must be a positive integer", index)
        return False

    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        logger.error("Record %s: 'prompt' must be a non-empty string", index)
        return False

    topic = record.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        logger.error("Record %s: 'topic' must be a non-empty string", index)
        return False

    language = record.get("language")
    if language not in ALLOWED_LANGUAGES:
        logger.error(
            "Record %s: 'language' must be one of %s", index, sorted(ALLOWED_LANGUAGES)
        )
        return False

    reference_outputs = record.get("reference_outputs")
    if not isinstance(reference_outputs, dict):
        logger.error("Record %s: 'reference_outputs' must be an object", index)
        return False

    parallel = reference_outputs.get("parallel")
    if not isinstance(parallel, int) or parallel < 0:
        logger.error(
            "Record %s: 'reference_outputs.parallel' must be a non-negative integer",
            index,
        )
        return False

    return True


def get_or_create_dataset(client: Client, dataset_name: str):
    """Fetch existing dataset or create a new one."""
    try:
        existing = list(client.list_datasets(dataset_name=dataset_name))
    except Exception:
        logger.exception("Failed to list datasets")
        raise

    if existing:
        dataset = existing[0]
        logger.info("Using existing dataset: %s (id: %s)", dataset_name, dataset.id)
        return dataset

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Supervisor parallelism evaluation dataset for Panda_Dive",
        )
        logger.info("Created dataset: %s (id: %s)", dataset_name, dataset.id)
        return dataset
    except Exception:
        logger.exception("Failed to create dataset")
        raise


def collect_existing_record_ids(client: Client, dataset_id: str) -> set[int]:
    """Collect record ids from existing dataset examples."""
    existing_ids: set[int] = set()
    try:
        for example in client.list_examples(dataset_id=dataset_id):
            metadata = example.metadata or {}
            record_id = metadata.get("id")
            if isinstance(record_id, int):
                existing_ids.add(record_id)
    except Exception:
        logger.exception("Failed to list existing examples")
        raise
    return existing_ids


def build_example(record: dict[str, Any], dataset_id: str) -> Example:
    """Build a LangSmith Example from a record."""
    return Example(
        id=uuid.uuid4(),
        dataset_id=dataset_id,
        inputs={"messages": [{"role": "user", "content": record["prompt"]}]},
        outputs={},
        metadata={
            "id": record["id"],
            "topic": record["topic"],
            "language": record["language"],
        },
        reference_outputs={"parallel": record["reference_outputs"]["parallel"]},
    )


def create_examples(
    client: Client, dataset_id: str, records: list[dict[str, Any]]
) -> int:
    """Create examples for dataset, skipping duplicates."""
    existing_ids = collect_existing_record_ids(client, dataset_id)
    created = 0

    for record in records:
        record_id = record["id"]
        if record_id in existing_ids:
            logger.info("Skipping existing record id=%s", record_id)
            continue

        example = build_example(record, dataset_id)
        try:
            # Build outputs with reference data for the supervisor parallelism evaluation
            outputs = {"parallel": record["reference_outputs"]["parallel"]}
            client.create_example(
                dataset_id=dataset_id,
                inputs=example.inputs,
                outputs=outputs,
                metadata=example.metadata,
            )
        except Exception:
            logger.exception("Failed to create example id=%s", record_id)
            raise
        created += 1

    return created


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Create LangSmith dataset for supervisor parallelism evaluation."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Panda_Dive: Supervisor Parallelism",
        help="Name for the LangSmith dataset",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to JSONL file with dataset records",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = create_parser()
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.warning("LANGSMITH_API_KEY not set. LangSmith calls may fail.")

    source_path = Path(args.source)
    if not source_path.exists():
        logger.error("Source file not found: %s", source_path)
        return 1

    try:
        records = load_jsonl(source_path)
    except ValueError as exc:
        logger.error("Failed to load JSONL: %s", exc)
        return 1

    if not records:
        logger.error("No records found in source file")
        return 1

    valid_records: list[dict[str, Any]] = []
    for i, record in enumerate(records, 1):
        if validate_record(record, i):
            valid_records.append(record)
        else:
            logger.warning("Skipping invalid record %s", i)

    if not valid_records:
        logger.error("No valid records after validation")
        return 1

    client = Client()
    try:
        dataset = get_or_create_dataset(client, args.dataset_name)
        created = create_examples(client, str(dataset.id), valid_records)
    except Exception:
        return 1

    logger.info("=" * 60)
    logger.info("Dataset creation complete")
    logger.info("  Dataset name: %s", args.dataset_name)
    logger.info("  Dataset ID: %s", dataset.id)
    logger.info("  Records total: %s", len(valid_records))
    logger.info("  Records created: %s", created)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
