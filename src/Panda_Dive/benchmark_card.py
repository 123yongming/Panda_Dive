"""Utilities for generating benchmark card summaries from LangSmith runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark metrics."""

    total_runs: int
    successful_runs: int
    success_rate: float
    latency_avg_seconds: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    total_cost: float | None
    avg_cost_per_run: float | None
    quality_metrics: dict[str, float]


def _percentile(values: list[float], ratio: float) -> float:
    """Compute percentile for a sorted list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int((len(values) - 1) * ratio)
    return values[idx]


def _extract_latency_seconds(run: Any) -> float | None:
    """Extract latency in seconds from a run."""
    start = getattr(run, "start_time", None)
    end = getattr(run, "end_time", None)
    if not start or not end:
        return None
    return max((end - start).total_seconds(), 0.0)


def _extract_quality_metrics(runs: list[Any]) -> dict[str, float]:
    """Aggregate evaluator metrics from feedback_stats."""
    metric_values: dict[str, list[float]] = {}

    for run in runs:
        feedback_stats = getattr(run, "feedback_stats", None) or {}
        if not isinstance(feedback_stats, dict):
            continue
        for key, value in feedback_stats.items():
            score: float | None = None
            if isinstance(value, dict):
                for field in ("avg", "mean", "score", "value"):
                    candidate = value.get(field)
                    if isinstance(candidate, int | float):
                        score = float(candidate)
                        break
            elif isinstance(value, int | float):
                score = float(value)

            if score is None:
                continue
            metric_values.setdefault(str(key), []).append(score)

    return {key: mean(values) for key, values in metric_values.items()}


def summarize_runs(runs: list[Any]) -> BenchmarkSummary:
    """Summarize quality, cost, and latency metrics from runs."""
    total_runs = len(runs)
    successful_runs = 0
    latencies: list[float] = []
    costs: list[float] = []

    for run in runs:
        final_report = (getattr(run, "outputs", None) or {}).get("final_report", "")
        if isinstance(final_report, str) and not final_report.startswith(
            ("Error:", "Error generating final report:")
        ):
            successful_runs += 1

        latency = _extract_latency_seconds(run)
        if latency is not None:
            latencies.append(latency)

        total_cost = getattr(run, "total_cost", None)
        if isinstance(total_cost, int | float):
            costs.append(float(total_cost))

    sorted_latencies = sorted(latencies)
    latency_avg = mean(latencies) if latencies else 0.0
    latency_p50 = _percentile(sorted_latencies, 0.50)
    latency_p95 = _percentile(sorted_latencies, 0.95)
    success_rate = (successful_runs / total_runs) if total_runs else 0.0

    total_cost_value = round(sum(costs), 6) if costs else None
    avg_cost_value = round(sum(costs) / total_runs, 6) if costs and total_runs else None

    return BenchmarkSummary(
        total_runs=total_runs,
        successful_runs=successful_runs,
        success_rate=success_rate,
        latency_avg_seconds=latency_avg,
        latency_p50_seconds=latency_p50,
        latency_p95_seconds=latency_p95,
        total_cost=total_cost_value,
        avg_cost_per_run=avg_cost_value,
        quality_metrics=_extract_quality_metrics(runs),
    )


def _format_float(value: float | None, digits: int = 4) -> str:
    """Format float values for markdown output."""
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_markdown_card(
    project_name: str,
    dataset_name: str,
    model_name: str,
    summary: BenchmarkSummary,
    generated_at: str | None = None,
) -> str:
    """Build benchmark card markdown."""
    now = generated_at or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    quality_lines = []
    for key, value in sorted(summary.quality_metrics.items()):
        quality_lines.append(f"- `{key}`: {_format_float(value, 4)}")
    if not quality_lines:
        quality_lines = ["- No evaluator metrics found in run feedback."]

    return f"""# Panda_Dive Benchmark Card

## Metadata
- Generated at: {now}
- LangSmith project: `{project_name}`
- Dataset: `{dataset_name}`
- Model: `{model_name}`
- Total runs: {summary.total_runs}

## Quality
{chr(10).join(quality_lines)}
- Success rate: {_format_float(summary.success_rate * 100, 2)}%
- Successful runs: {summary.successful_runs}/{summary.total_runs}

## Cost
- Total estimated cost (USD): {_format_float(summary.total_cost, 6)}
- Average cost per run (USD): {_format_float(summary.avg_cost_per_run, 6)}

## Latency
- Average latency (s): {_format_float(summary.latency_avg_seconds, 3)}
- P50 latency (s): {_format_float(summary.latency_p50_seconds, 3)}
- P95 latency (s): {_format_float(summary.latency_p95_seconds, 3)}
"""
