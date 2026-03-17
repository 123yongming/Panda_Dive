"""Integration tests for memory.store functions with SQLite backend."""
import shutil
import uuid
from pathlib import Path

import pytest

from Panda_Dive.memory.schemas import MemoryEpisode, MemoryFact
from Panda_Dive.memory.store import (
    resolve_namespace,
    search_episodes,
    search_facts,
    upsert_episode,
    upsert_fact,
)


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


def _build_config(db_path: str, template: str = "memory.owner.{owner}") -> dict:
    return {
        "configurable": {
            "memory_enabled": True,
            "memory_backend": "sqlite",
            "memory_sqlite_path": db_path,
            "memory_embedding_enabled": False,
            "memory_namespace_template": template,
            "memory_retrieval_top_k": 8,
            "memory_max_facts_per_namespace": 500,
            "owner": "owner_sqlite",
            "thread_id": "thread_1",
        }
    }


def _build_fact() -> MemoryFact:
    return MemoryFact(
        id="fact_sqlite_1",
        namespace="",
        fact_type="knowledge",
        content="SQLite backend keeps LangMem-reconciled memory persistent across runs.",
        confidence=0.42,
        novelty=1.0,
        source_urls=[],
        source_run_id="run_1",
        source_message_ids=["msg_1"],
        created_at="2026-03-04T00:00:00Z",
        updated_at="2026-03-04T00:00:00Z",
    )


def _build_episode() -> MemoryEpisode:
    return MemoryEpisode(
        id="episode_sqlite_1",
        namespace="",
        topic="sqlite memory backend",
        summary="A stable summary persisted into sqlite memory backend.",
        key_findings=["finding a", "finding b"],
        citations=[],
        quality_score=0.88,
        created_at="2026-03-04T00:00:00Z",
    )


@pytest.fixture(name="local_tmp_path")
def _local_tmp_path_fixture():
    """Create a writable temporary directory under repository root."""
    root = Path(".pytest_tmp")
    root.mkdir(exist_ok=True)
    tmp = root / f"sqlite_memory_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.anyio
async def test_memory_store_fact_flow_with_sqlite_backend(local_tmp_path, monkeypatch):
    """upsert_fact and search_facts should work end-to-end with sqlite backend."""
    monkeypatch.setattr(
        "Panda_Dive.memory.store._get_namespace_template_cls",
        lambda: _FakeNamespaceTemplate,
    )
    db_path = str(local_tmp_path / "memory.sqlite3")
    config = _build_config(db_path)

    accepted = await upsert_fact(_build_fact(), config)
    assert accepted is True

    facts = await search_facts("persistent", config, top_k=5)
    assert len(facts) >= 1
    assert "LangMem-reconciled" in facts[0].content


@pytest.mark.anyio
async def test_memory_store_topic_template_isolation_with_sqlite_backend(local_tmp_path, monkeypatch):
    """Topic-hash templates should isolate fact and episode searches."""
    monkeypatch.setattr(
        "Panda_Dive.memory.store._get_namespace_template_cls",
        lambda: _FakeNamespaceTemplate,
    )
    db_path = str(local_tmp_path / "memory.sqlite3")
    config = _build_config(
        db_path,
        template="memory.owner.{owner}.topic.{topic_hash}",
    )
    namespace = resolve_namespace(config, topic="sqlite memory backend")

    accepted_fact = await upsert_fact(_build_fact(), config, namespace=namespace)
    accepted_episode = await upsert_episode(
        _build_episode(),
        config,
        namespace=namespace,
    )
    assert accepted_fact is True
    assert accepted_episode is True

    facts = await search_facts(
        "persistent",
        config,
        top_k=5,
        topic="sqlite memory backend",
    )
    episodes = await search_episodes(
        "stable summary",
        config,
        top_k=5,
        topic="sqlite memory backend",
    )
    missing_topic_facts = await search_facts(
        "persistent",
        config,
        top_k=5,
        topic="other topic",
    )

    assert len(facts) >= 1
    assert len(episodes) >= 1
    assert missing_topic_facts == []
