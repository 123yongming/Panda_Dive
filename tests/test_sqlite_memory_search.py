"""Tests for lexical and vector retrieval behavior in SQLite memory backend."""

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
async def test_sqlite_memory_search_rrf_supports_vector_only_recall(local_tmp_path):
    """Search should return vector-only matches when lexical path has no hits."""

    async def _embed(text: str) -> list[float] | None:
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0]
        return [1.0, 0.0]

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
        embed_text=_embed,
    )
    namespace = ("memory", "owner", "u1")
    await store.aput(
        namespace,
        "fact_alpha",
        {"kind": "fact", "content": "alpha memory retrieval strategy"},
    )
    await store.aput(
        namespace,
        "fact_beta",
        {"kind": "fact", "content": "beta memory retrieval strategy"},
    )

    results = await store.asearch(namespace, query="semantic query only", limit=2)
    assert len(results) >= 1
    assert results[0].key == "fact_alpha"
    assert (results[0].score or 0.0) > 0.0


@pytest.mark.anyio
async def test_sqlite_memory_search_falls_back_to_lexical_when_embedding_unavailable(local_tmp_path):
    """Search should still work when embedding function returns None."""

    async def _missing_embed(_text: str) -> list[float] | None:
        return None

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
        embed_text=_missing_embed,
    )
    namespace = ("memory", "owner", "u1")
    await store.aput(
        namespace,
        "fact_alpha",
        {"kind": "fact", "content": "alpha memory retrieval strategy"},
    )
    await store.aput(
        namespace,
        "fact_beta",
        {"kind": "fact", "content": "beta memory retrieval strategy"},
    )

    results = await store.asearch(namespace, query="beta", limit=2)
    assert len(results) >= 1
    assert results[0].key == "fact_beta"


@pytest.mark.anyio
async def test_sqlite_memory_search_preserves_bm25_rank_for_fts_hits(local_tmp_path):
    """FTS lexical ranking should prefer better BM25 matches over newer partial hits."""
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
    await store.aput(
        namespace,
        "fact_exact",
        {
            "kind": "fact",
            "content": "alpha strategy",
            "updated_at": "2026-03-01T00:00:00Z",
        },
    )
    await store.aput(
        namespace,
        "fact_partial",
        {
            "kind": "fact",
            "content": "alpha retrieval planning strategy noise noise noise",
            "updated_at": "2026-03-02T00:00:00Z",
        },
    )

    await store._ensure_initialized()
    assert store._fts_enabled is True

    results = await store.asearch(namespace, query="alpha strategy", limit=2)

    assert [item.key for item in results] == ["fact_exact", "fact_partial"]
