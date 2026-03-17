"""Tests for memory store utilities."""

import hashlib
from datetime import datetime, timezone

import pytest
from langgraph.store.base import Item, SearchItem

from Panda_Dive.memory.schemas import MemoryFact
from Panda_Dive.memory.store import resolve_namespace, search_facts, upsert_fact


class _FakeNamespaceTemplate:
    """Small test double that mirrors LangMem segment substitution."""

    def __init__(self, template):
        self.template = tuple(template)

    def __call__(self, config):
        configurable = config.get("configurable", {})
        resolved = []
        for part in self.template:
            if part.startswith("{") and part.endswith("}"):
                resolved.append(str(configurable[part[1:-1]]))
            else:
                resolved.append(part)
        return tuple(resolved)


class _FakeStore:
    """Simple in-memory async store for tests."""

    def __init__(self):
        self._data: dict[tuple[str, ...], dict[str, dict]] = {}

    async def aput(self, namespace, key, value, index=None, ttl=None):
        _ = index, ttl
        self._data.setdefault(namespace, {})[key] = value

    async def aget(self, namespace, key, refresh_ttl=None):
        _ = refresh_ttl
        value = self._data.get(namespace, {}).get(key)
        if value is None:
            return None
        now = datetime.now(timezone.utc)
        return Item(
            value=value,
            key=key,
            namespace=namespace,
            created_at=now,
            updated_at=now,
        )

    async def adelete(self, namespace, key):
        self._data.get(namespace, {}).pop(key, None)

    async def asearch(
        self,
        namespace_prefix,
        query=None,
        filter=None,
        limit=10,
        offset=0,
        refresh_ttl=None,
    ):
        _ = refresh_ttl
        query_text = (query or "").lower()
        items: list[SearchItem] = []
        now = datetime.now(timezone.utc)
        for namespace, values in self._data.items():
            if namespace[: len(namespace_prefix)] != namespace_prefix:
                continue
            for key, value in values.items():
                if filter and any(value.get(k) != v for k, v in filter.items()):
                    continue
                content = str(value.get("content", "")).lower()
                summary = str(value.get("summary", "")).lower()
                topic = str(value.get("topic", "")).lower()
                if query_text and query_text not in content and query_text not in summary and query_text not in topic:
                    continue
                items.append(
                    SearchItem(
                        namespace=namespace,
                        key=key,
                        value=value,
                        created_at=now,
                        updated_at=now,
                        score=1.0,
                    )
                )
        return items[offset : offset + limit]


def _build_config(template: str = "memory.owner.{owner}") -> dict:
    return {
        "configurable": {
            "memory_enabled": True,
            "memory_namespace_template": template,
            "memory_retrieval_top_k": 8,
            "memory_max_facts_per_namespace": 500,
            "owner": "owner_1",
            "thread_id": "thread_1",
        }
    }


def _build_fact(content: str, confidence: float = 0.9) -> MemoryFact:
    return MemoryFact(
        id="fact_a",
        namespace="",
        fact_type="knowledge",
        content=content,
        confidence=confidence,
        novelty=1.0,
        source_urls=[],
        source_run_id="run_1",
        source_message_ids=["msg_1"],
        created_at="2026-03-04T00:00:00Z",
        updated_at="2026-03-04T00:00:00Z",
    )


@pytest.mark.anyio
async def test_upsert_fact_does_not_apply_legacy_rejection(monkeypatch):
    """Store writes should no longer reject low-confidence uncited facts."""
    monkeypatch.setattr(
        "Panda_Dive.memory.store._get_namespace_template_cls",
        lambda: _FakeNamespaceTemplate,
    )
    store = _FakeStore()

    accepted = await upsert_fact(
        _build_fact("Low confidence but should still be stored.", confidence=0.2),
        _build_config(),
        store=store,
    )

    assert accepted is True


@pytest.mark.anyio
async def test_search_facts_returns_written_fact(monkeypatch):
    """Search should return fact after successful upsert into store."""
    monkeypatch.setattr(
        "Panda_Dive.memory.store._get_namespace_template_cls",
        lambda: _FakeNamespaceTemplate,
    )
    store = _FakeStore()
    fact = _build_fact("Panda_Dive uses LangMem-backed reconciliation.")
    accepted = await upsert_fact(fact, _build_config(), store=store)
    assert accepted is True

    facts = await search_facts("langmem", _build_config(), store=store, top_k=5)
    assert len(facts) == 1
    assert facts[0].content.startswith("Panda_Dive")


def test_resolve_namespace_uses_template_variables(monkeypatch):
    """Template-based namespaces should resolve owner, thread, and topic hash."""
    monkeypatch.setattr(
        "Panda_Dive.memory.store._get_namespace_template_cls",
        lambda: _FakeNamespaceTemplate,
    )
    config = {
        "configurable": {
            "memory_enabled": True,
            "memory_namespace_template": "memory.owner.{owner}.thread.{thread_id}.topic.{topic_hash}",
            "thread_id": "thread_9",
        },
        "metadata": {"owner": "owner_meta"},
    }

    namespace = resolve_namespace(config, topic="Memory Topic")

    assert namespace[:6] == ("memory", "owner", "owner_meta", "thread", "thread_9", "topic")
    assert namespace[-1] == hashlib.sha1(b"Memory Topic").hexdigest()[:12]
