"""Memory metrics helpers."""

from dataclasses import dataclass

from .injector import estimate_token_count


@dataclass
class MemoryMetrics:
    """Memory quality and runtime metrics for a single turn."""

    injection_tokens: int = 0
    retrieved_facts: int = 0
    retrieved_episodes: int = 0
    accepted_writes: int = 0
    rejected_writes: int = 0
    conflicts: int = 0


def build_metrics(
    injection_block: str,
    *,
    retrieved_facts: int,
    retrieved_episodes: int,
    accepted_writes: int,
    rejected_writes: int,
    conflicts: int,
) -> MemoryMetrics:
    """Build a MemoryMetrics instance from counters."""
    return MemoryMetrics(
        injection_tokens=estimate_token_count(injection_block),
        retrieved_facts=retrieved_facts,
        retrieved_episodes=retrieved_episodes,
        accepted_writes=accepted_writes,
        rejected_writes=rejected_writes,
        conflicts=conflicts,
    )
