"""Tests for memory prompt injection."""

from Panda_Dive.memory.injector import (
    build_memory_injection_block,
    estimate_token_count,
)
from Panda_Dive.memory.schemas import MemoryEpisode, MemoryFact


def test_build_memory_injection_block_respects_budget():
    """Injector should keep assembled memory block within token budget."""
    facts = [
        MemoryFact(
            id=f"fact_{index}",
            namespace="memory.owner.default",
            fact_type="knowledge",
            content=f"Fact {index} content about retrieval and context quality.",
            confidence=0.9,
            novelty=0.8,
            source_urls=["https://example.com"],
            source_run_id="run_1",
            source_message_ids=["msg_1"],
            created_at="2026-03-04T00:00:00Z",
            updated_at="2026-03-04T00:00:00Z",
            rank_score=1.0 - index * 0.01,
        )
        for index in range(20)
    ]
    episodes = [
        MemoryEpisode(
            id="ep_1",
            namespace="memory.owner.default",
            topic="memory architecture",
            summary="Episode summary.",
            key_findings=["f1", "f2"],
            citations=["https://example.com/ep"],
            quality_score=0.91,
            created_at="2026-03-04T00:00:00Z",
            rank_score=0.9,
        )
    ]
    block = build_memory_injection_block(
        facts=facts,
        episodes=episodes,
        preferences=["Prefer evidence-backed claims."],
        max_tokens=120,
    )
    assert "<memory_context>" in block
    assert "Facts" in block
    assert estimate_token_count(block) <= 130
