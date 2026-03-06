"""Tests for memory retriever behavior after heuristic rerank removal."""

import pytest

from Panda_Dive.memory.retriever import retrieve_memory_bundle
from Panda_Dive.memory.schemas import MemoryFact


def _fact(fact_id: str, content: str, fact_type: str = "knowledge") -> MemoryFact:
    return MemoryFact(
        id=fact_id,
        namespace="memory.owner.default",
        fact_type=fact_type,
        content=content,
        confidence=0.5,
        novelty=0.5,
        source_urls=["https://example.com"],
        source_run_id="run_1",
        source_message_ids=["msg_1"],
        created_at="2026-03-04T00:00:00Z",
        updated_at="2026-03-04T00:00:00Z",
        rank_score=0.0,
    )


@pytest.mark.anyio
async def test_retrieve_memory_bundle_keeps_store_order(monkeypatch):
    """Retriever should keep store ordering and avoid heuristic reordering."""
    ordered = [
        _fact("f1", "first from store"),
        _fact("f2", "second from store", fact_type="preference"),
    ]

    async def _search_facts(*args, **kwargs):
        _ = args, kwargs
        return ordered

    async def _search_episodes(*args, **kwargs):
        _ = args, kwargs
        return []

    monkeypatch.setattr("Panda_Dive.memory.retriever.search_facts", _search_facts)
    monkeypatch.setattr("Panda_Dive.memory.retriever.search_episodes", _search_episodes)

    bundle = await retrieve_memory_bundle(
        query="any",
        task_context="ignored",
        config={"configurable": {"memory_enabled": True, "memory_retrieval_top_k": 2}},
    )

    assert [item.id for item in bundle.facts] == ["f1", "f2"]
    assert bundle.preferences == ["second from store"]

