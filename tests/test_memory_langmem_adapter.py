"""Tests for the LangMem extraction adapter."""

import hashlib
from collections import namedtuple
from datetime import datetime, timezone

import pytest
from langgraph.store.base import Item, SearchItem

from Panda_Dive.memory import persist_research_memory
from Panda_Dive.memory.extractor import ResearchEpisodeMemory, ResearchFactMemory

FakeExtractedMemory = namedtuple("FakeExtractedMemory", ["id", "content"])


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
    """Simple in-memory async store for integration-style tests."""

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


class _FakeManager:
    """Simple LangMem manager test double."""

    def __init__(self, results):
        self._results = results

    async def ainvoke(self, payload, config=None):
        _ = payload, config
        return list(self._results)


class _FakeRemoveDoc:
    """Test double for trustcall RemoveDoc."""

    def __init__(self, json_doc_id: str):
        self.json_doc_id = json_doc_id

    def __repr_name__(self):
        return "RemoveDoc"


def _build_config(template: str = "memory.owner.{owner}.topic.{topic_hash}") -> dict:
    return {
        "configurable": {
            "memory_enabled": True,
            "memory_backend": "langgraph_store",
            "memory_namespace_template": template,
            "memory_retrieval_top_k": 8,
            "memory_max_facts_per_namespace": 500,
            "compression_model": "deepseek-chat",
            "compression_model_max_tokens": 2048,
            "max_structured_output_retries": 0,
            "owner": "owner_1",
            "thread_id": "thread_1",
        }
    }


def test_get_memory_manager_builds_base_chat_model(monkeypatch):
    """LangMem adapter should pass a base chat model, not a retry wrapper."""
    captured = {}
    sentinel_model = object()
    sentinel_manager = object()

    def _fake_init_chat_model(**kwargs):
        captured["init_kwargs"] = kwargs
        return sentinel_model

    monkeypatch.setattr(
        "Panda_Dive.memory.extractor.init_chat_model",
        _fake_init_chat_model,
    )
    monkeypatch.setattr(
        "Panda_Dive.memory.extractor.get_api_key_for_model",
        lambda model_name, config: "test-key",
    )

    def _fake_factory():
        def _create_manager(model, **kwargs):
            captured["manager_model"] = model
            captured["manager_kwargs"] = kwargs
            return sentinel_manager

        return _create_manager

    monkeypatch.setattr(
        "Panda_Dive.memory.extractor._get_memory_manager_factory",
        _fake_factory,
    )
    monkeypatch.setattr(
        "Panda_Dive.memory.extractor._MANAGER_CACHE",
        {},
    )

    manager = __import__(
        "Panda_Dive.memory.extractor",
        fromlist=["get_memory_manager"],
    ).get_memory_manager(_build_config())

    assert manager is sentinel_manager
    assert captured["manager_model"] is sentinel_model
    assert captured["init_kwargs"]["model"] == "deepseek-chat"


@pytest.mark.anyio
async def test_persist_research_memory_inserts_fact_and_episode(monkeypatch):
    """Persist flow should map LangMem insertions into current fact and episode rows."""
    monkeypatch.setattr(
        "Panda_Dive.memory.store._get_namespace_template_cls",
        lambda: _FakeNamespaceTemplate,
    )
    store = _FakeStore()
    manager = _FakeManager(
        [
            FakeExtractedMemory(
                "new_fact",
                ResearchFactMemory(
                    content="LangMem should govern durable research facts.",
                    confidence=0.91,
                ),
            ),
            FakeExtractedMemory(
                "new_episode",
                ResearchEpisodeMemory(
                    topic="langmem integration",
                    summary="A run summary for the LangMem integration work.",
                    key_findings=["replace heuristic extraction", "keep SQLite retrieval"],
                    quality_score=0.86,
                ),
            ),
        ]
    )
    monkeypatch.setattr(
        "Panda_Dive.memory.extractor.get_memory_manager",
        lambda _config: manager,
    )
    monkeypatch.setattr(
        "Panda_Dive.memory.store.get_store",
        lambda: store,
    )

    await persist_research_memory(
        topic="langmem integration",
        compressed_research="LangMem now manages extracted memories.",
        raw_notes="Source https://example.com/langmem",
        config=_build_config(),
        source_run_id="run_1",
        source_message_ids=["msg_1"],
    )

    namespace = next(iter(store._data))
    payloads = store._data[namespace]
    assert len(payloads) == 2
    fact_payload = next(value for value in payloads.values() if value["kind"] == "fact")
    episode_payload = next(value for value in payloads.values() if value["kind"] == "episode")
    assert fact_payload["content"] == "LangMem should govern durable research facts."
    assert fact_payload["source_urls"] == ["https://example.com/langmem"]
    assert episode_payload["topic"] == "langmem integration"
    assert episode_payload["citations"] == ["https://example.com/langmem"]


@pytest.mark.anyio
async def test_persist_research_memory_updates_and_deletes_existing_records(monkeypatch):
    """Persist flow should update existing rows in place and delete RemoveDoc targets."""
    monkeypatch.setattr(
        "Panda_Dive.memory.store._get_namespace_template_cls",
        lambda: _FakeNamespaceTemplate,
    )
    store = _FakeStore()
    topic_hash = hashlib.sha1(b"langmem integration").hexdigest()[:12]
    namespace = ("memory", "owner", "owner_1", "topic", topic_hash)
    await store.aput(
        namespace,
        "fact_existing",
        {
            "kind": "fact",
            "id": "fact_existing",
            "namespace": ".".join(namespace),
            "fact_type": "knowledge",
            "content": "Old fact",
            "confidence": 0.7,
            "novelty": 1.0,
            "source_urls": ["https://example.com/old"],
            "source_run_id": "run_old",
            "source_message_ids": ["msg_old"],
            "created_at": "2026-03-04T00:00:00Z",
            "updated_at": "2026-03-04T00:00:00Z",
            "status": "active",
        },
    )
    await store.aput(
        namespace,
        "episode_existing",
        {
            "kind": "episode",
            "id": "episode_existing",
            "namespace": ".".join(namespace),
            "topic": "langmem integration",
            "summary": "Old episode",
            "key_findings": ["old"],
            "citations": ["https://example.com/old"],
            "quality_score": 0.5,
            "created_at": "2026-03-04T00:00:00Z",
        },
    )
    manager = _FakeManager(
        [
            FakeExtractedMemory(
                "fact_existing",
                ResearchFactMemory(
                    content="Updated fact from LangMem reconciliation.",
                    confidence=0.95,
                ),
            ),
            FakeExtractedMemory(
                "episode_existing",
                _FakeRemoveDoc("episode_existing"),
            ),
        ]
    )
    monkeypatch.setattr(
        "Panda_Dive.memory.extractor.get_memory_manager",
        lambda _config: manager,
    )
    monkeypatch.setattr(
        "Panda_Dive.memory.store.get_store",
        lambda: store,
    )

    await persist_research_memory(
        topic="langmem integration",
        compressed_research="Updated reconciliation results.",
        raw_notes="Source https://example.com/new",
        config=_build_config(),
        source_run_id="run_new",
        source_message_ids=["msg_new"],
    )

    payloads = store._data[namespace]
    assert payloads["fact_existing"]["content"] == "Updated fact from LangMem reconciliation."
    assert payloads["fact_existing"]["source_urls"] == [
        "https://example.com/old",
        "https://example.com/new",
    ]
    assert payloads["fact_existing"]["source_message_ids"] == ["msg_old", "msg_new"]
    assert "episode_existing" not in payloads
