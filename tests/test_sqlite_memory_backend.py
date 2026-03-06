"""Tests for SQLite memory backend CRUD and prefix search behavior."""

import shutil
import uuid
from pathlib import Path

import pytest

from Panda_Dive.memory.sqlite_backend import SQLiteMemoryStore


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
async def test_sqlite_memory_store_crud(local_tmp_path):
    """SQLite backend should support put/get/search/delete flow."""
    store = SQLiteMemoryStore(
        db_path=str(local_tmp_path / "memory.sqlite3"),
        journal_mode="WAL",
        busy_timeout_ms=5000,
        search_candidates=100,
        ann_enabled=False,
        ann_max_elements=10000,
        ann_candidates=50,
        rrf_k=60,
        rrf_candidate_max=200,
        embed_text=None,
    )
    namespace = ("memory", "owner", "u1")
    payload = {
        "kind": "fact",
        "content": "Memory injection should respect prompt budget.",
        "confidence": 0.92,
    }
    await store.aput(namespace, "fact_1", payload, index=["content"])

    item = await store.aget(namespace, "fact_1")
    assert item is not None
    assert item.value["content"].startswith("Memory injection")

    results = await store.asearch(
        ("memory", "owner", "u1"),
        query="prompt budget",
        filter={"kind": "fact"},
        limit=5,
    )
    assert len(results) == 1
    assert results[0].key == "fact_1"

    await store.adelete(namespace, "fact_1")
    deleted = await store.aget(namespace, "fact_1")
    assert deleted is None


@pytest.mark.anyio
async def test_sqlite_memory_search_respects_namespace_prefix(local_tmp_path):
    """Search should include only records under the specified namespace prefix."""
    store = SQLiteMemoryStore(
        db_path=str(local_tmp_path / "memory.sqlite3"),
        journal_mode="WAL",
        busy_timeout_ms=5000,
        search_candidates=100,
        ann_enabled=False,
        ann_max_elements=10000,
        ann_candidates=50,
        rrf_k=60,
        rrf_candidate_max=200,
        embed_text=None,
    )
    await store.aput(
        ("memory", "owner", "u1", "thread", "t1"),
        "fact_a",
        {"kind": "fact", "content": "Owner u1 fact"},
    )
    await store.aput(
        ("memory", "owner", "u2", "thread", "t1"),
        "fact_b",
        {"kind": "fact", "content": "Owner u2 fact"},
    )

    u1_results = await store.asearch(("memory", "owner", "u1"), query="fact", limit=10)
    u2_results = await store.asearch(("memory", "owner", "u2"), query="fact", limit=10)

    assert len(u1_results) == 1
    assert len(u2_results) == 1
    assert u1_results[0].key == "fact_a"
    assert u2_results[0].key == "fact_b"
