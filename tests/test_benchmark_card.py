"""Tests for benchmark card metric aggregation."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from Panda_Dive.benchmark_card import build_markdown_card, summarize_runs


def _make_run(
    *,
    final_report: str,
    latency_seconds: int,
    total_cost: float | None,
    feedback_stats: dict | None,
):
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=latency_seconds)
    return SimpleNamespace(
        outputs={"final_report": final_report},
        start_time=start,
        end_time=end,
        total_cost=total_cost,
        feedback_stats=feedback_stats or {},
    )


def test_summarize_runs_computes_success_latency_and_cost() -> None:
    """Test that summarize_runs correctly computes metrics from multiple runs."""
    runs = [
        _make_run(
            final_report="ok",
            latency_seconds=10,
            total_cost=0.1,
            feedback_stats={"overall_quality": {"avg": 0.8}},
        ),
        _make_run(
            final_report="Error: timeout",
            latency_seconds=20,
            total_cost=0.2,
            feedback_stats={"overall_quality": {"avg": 0.6}},
        ),
    ]

    summary = summarize_runs(runs)

    assert summary.total_runs == 2
    assert summary.successful_runs == 1
    assert summary.success_rate == 0.5
    assert summary.latency_avg_seconds == 15.0
    assert summary.total_cost == 0.3
    assert summary.avg_cost_per_run == 0.15
    assert summary.quality_metrics["overall_quality"] == 0.7


def test_summarize_runs_treats_final_report_generation_error_as_failure() -> None:
    """Test that 'Error generating final report:' is treated as failure."""
    runs = [
        _make_run(
            final_report="Error generating final report: token limit exceeded",
            latency_seconds=3,
            total_cost=0.01,
            feedback_stats=None,
        )
    ]

    summary = summarize_runs(runs)

    assert summary.total_runs == 1
    assert summary.successful_runs == 0
    assert summary.success_rate == 0.0


def test_build_markdown_card_contains_core_sections() -> None:
    """Test that markdown card contains all required sections."""
    runs = [
        _make_run(
            final_report="ok",
            latency_seconds=5,
            total_cost=0.05,
            feedback_stats={"overall_quality": {"avg": 0.9}},
        )
    ]
    summary = summarize_runs(runs)
    card = build_markdown_card(
        project_name="demo-project",
        dataset_name="Demo Dataset",
        model_name="openai:gpt-4o",
        summary=summary,
        generated_at="2026-03-03",
    )

    assert "# Panda_Dive Benchmark Card" in card
    assert "Quality" in card
    assert "Cost" in card
    assert "Latency" in card
    assert "overall_quality" in card
