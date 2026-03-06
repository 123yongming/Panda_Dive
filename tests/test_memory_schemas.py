"""Tests for memory schemas."""

from Panda_Dive.memory.schemas import MemoryCandidate, MemoryEpisode, MemoryFact


def test_memory_fact_defaults_are_applied():
    """Fact model should apply default status and ttl fields."""
    fact = MemoryFact(
        id="fact_1",
        namespace="memory.owner.default",
        fact_type="knowledge",
        content="Panda_Dive uses LangGraph.",
        confidence=0.92,
        novelty=0.55,
        source_urls=["https://example.com"],
        source_run_id="run_1",
        source_message_ids=["msg_1"],
        created_at="2026-03-04T00:00:00Z",
        updated_at="2026-03-04T00:00:00Z",
    )
    assert fact.status == "active"
    assert fact.ttl_days is None


def test_memory_episode_schema():
    """Episode model should preserve provided topic and quality score."""
    episode = MemoryEpisode(
        id="ep_1",
        namespace="memory.owner.default",
        topic="multi-agent memory",
        summary="A run summary.",
        key_findings=["finding a", "finding b"],
        citations=["https://example.com/a"],
        quality_score=0.88,
        created_at="2026-03-04T00:00:00Z",
    )
    assert episode.quality_score == 0.88
    assert episode.topic == "multi-agent memory"


def test_memory_candidate_schema():
    """Candidate model should apply default novelty and active status."""
    candidate = MemoryCandidate(
        fact_type="knowledge",
        content="Use evidence-first memory extraction.",
        confidence=0.81,
        source_urls=["https://example.com/b"],
        source_message_ids=["msg_2"],
    )
    assert candidate.novelty == 1.0
    assert candidate.status == "active"
