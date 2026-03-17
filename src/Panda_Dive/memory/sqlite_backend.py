"""SQLite storage backend for memory persistence and retrieval."""

import asyncio
import json
import logging
import math
import sqlite3
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.store.base import Item, SearchItem

from ..configuration import Configuration
from .embedding import EmbeddingFunction, build_embedding_function
from .ranking import reciprocal_rank_fusion

ANN_META_ROW_ID = 1
ANN_TABLE_NAME = "memory_ann_index"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_datetime(timestamp: str) -> datetime:
    fixed = (timestamp or _utcnow_iso()).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(fixed)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _namespace_to_str(namespace: tuple[str, ...]) -> str:
    return ".".join(namespace)


def _namespace_from_str(namespace: str) -> tuple[str, ...]:
    if not namespace:
        return tuple()
    return tuple(part for part in namespace.split(".") if part)


def _value_matches_filter(value: dict[str, Any], filter_dict: dict[str, Any] | None) -> bool:
    if not filter_dict:
        return True
    return all(value.get(key) == expected for key, expected in filter_dict.items())


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0
    dot = sum(left * right for left, right in zip(first, second, strict=False))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return dot / (first_norm * second_norm)


def _fallback_lexical_score(query: str, content: str) -> float:
    query_tokens = {token for token in query.lower().split() if token}
    content_tokens = {token for token in content.lower().split() if token}
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    return overlap / len(query_tokens)


def _vector_to_blob(values: list[float]) -> bytes:
    return array("f", [float(item) for item in values]).tobytes()


def _item_id(namespace: str, key: str) -> str:
    return f"{namespace}\t{key}"


def _safe_updated_epoch(timestamp: str) -> float:
    try:
        return _parse_iso_datetime(timestamp).timestamp()
    except Exception:
        return 0.0


class SQLiteMemoryStore:
    """SQLite-based memory store with BM25 + ANN + RRF retrieval."""

    def __init__(
        self,
        *,
        db_path: str,
        journal_mode: str,
        busy_timeout_ms: int,
        search_candidates: int,
        ann_enabled: bool,
        ann_max_elements: int,
        ann_candidates: int,
        rrf_k: int,
        rrf_candidate_max: int,
        vector_weight: float | None = None,
        embed_text: EmbeddingFunction | None = None,
    ) -> None:
        """Initialize SQLite store runtime settings."""
        self._db_path = str(Path(db_path))
        self._journal_mode = journal_mode.upper() if journal_mode.upper() in {"WAL", "DELETE"} else "WAL"
        self._busy_timeout_ms = max(100, int(busy_timeout_ms))
        self._search_candidates = max(10, int(search_candidates))
        self._ann_enabled = bool(ann_enabled)
        self._ann_max_elements = max(1000, int(ann_max_elements))
        self._ann_candidates = max(10, int(ann_candidates))
        self._rrf_k = max(1, int(rrf_k))
        self._rrf_candidate_max = max(10, int(rrf_candidate_max))
        self._embed_text = embed_text

        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._initialized = False
        self._fts_enabled = True
        self._warned_fts_query = False
        self._warned_vector_weight_deprecated = False

        self._vectorlite_extension_path: str | None = None
        self._ann_extension_resolved = False
        self._ann_extension_warned = False
        self._ann_available = False
        self._ann_table_ready = False
        self._ann_backfill_done = False
        self._ann_dimension: int | None = None
        self._ann_dimension_warned = False
        self._ann_metadata_warned = False
        self._embedding_provider = ""
        self._embedding_model = ""

        if vector_weight is not None:
            logging.warning("memory_vector_weight is deprecated and ignored.")
            self._warned_vector_weight_deprecated = True

    def update_runtime_settings(
        self,
        configurable: Configuration,
        embed_text: EmbeddingFunction | None = None,
    ) -> None:
        """Update runtime tunables for this store instance."""
        self._search_candidates = max(10, configurable.memory_search_candidates)
        self._ann_enabled = bool(configurable.memory_ann_enabled)
        self._ann_max_elements = max(1000, configurable.memory_ann_max_elements)
        self._ann_candidates = max(10, configurable.memory_ann_candidates)
        self._rrf_k = max(1, configurable.memory_rrf_k)
        self._rrf_candidate_max = max(10, configurable.memory_rrf_candidate_max)
        self._embedding_provider = configurable.memory_embedding_provider
        self._embedding_model = configurable.memory_embedding_model
        if not self._warned_vector_weight_deprecated and configurable.memory_vector_weight is not None:
            logging.warning("memory_vector_weight is deprecated and ignored.")
            self._warned_vector_weight_deprecated = True
        if embed_text is not None:
            self._embed_text = embed_text

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _resolve_vectorlite_extension_path(self) -> str | None:
        if self._ann_extension_resolved:
            return self._vectorlite_extension_path
        self._ann_extension_resolved = True
        try:
            import vectorlite_py  # type: ignore

            self._vectorlite_extension_path = str(vectorlite_py.vectorlite_path())
        except Exception:
            self._vectorlite_extension_path = None
            if not self._ann_extension_warned:
                logging.warning("vectorlite extension unavailable; fallback to linear vector search.")
                self._ann_extension_warned = True
        return self._vectorlite_extension_path

    def _load_ann_extension_sync(self, connection: sqlite3.Connection) -> None:
        if not self._ann_enabled:
            self._ann_available = False
            return
        extension_path = self._resolve_vectorlite_extension_path()
        if not extension_path:
            self._ann_available = False
            return
        try:
            connection.enable_load_extension(True)
            connection.load_extension(extension_path)
            self._ann_available = True
        except Exception:
            self._ann_available = False
            if not self._ann_extension_warned:
                logging.warning("Failed to load vectorlite extension; fallback to linear vector search.")
                self._ann_extension_warned = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA journal_mode={self._journal_mode}")
        connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        connection.execute("PRAGMA synchronous=NORMAL")
        if self._ann_enabled:
            self._load_ann_extension_sync(connection)
        return connection

    def _initialize_sync(self) -> None:
        db_file = Path(self._db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT,
                    topic TEXT,
                    summary TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vector_ids (
                    vector_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    UNIQUE(namespace, key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vector_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    embedding_dim INTEGER NOT NULL,
                    embedding_model TEXT,
                    embedding_provider TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_items_kind_ns_updated
                ON memory_items (kind, namespace, updated_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_items_namespace
                ON memory_items (namespace)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_vector_ids_namespace
                ON memory_vector_ids (namespace)
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        namespace UNINDEXED,
                        key UNINDEXED,
                        kind UNINDEXED,
                        content,
                        topic,
                        summary
                    )
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
                logging.warning("SQLite FTS5 is unavailable. Search will fallback to LIKE.")
            self._initialize_ann_state_sync(connection)

    def _read_ann_dimension_from_meta_sync(self, connection: sqlite3.Connection) -> int | None:
        row = connection.execute(
            """
            SELECT embedding_dim, embedding_model, embedding_provider
            FROM memory_vector_meta
            WHERE id = ?
            """,
            (ANN_META_ROW_ID,),
        ).fetchone()
        if row is None:
            return None
        saved_model = str(row["embedding_model"] or "")
        saved_provider = str(row["embedding_provider"] or "")
        if (saved_model and saved_model != self._embedding_model) or (
            saved_provider and saved_provider != self._embedding_provider
        ):
            if not self._ann_metadata_warned:
                logging.warning("Embedding provider/model changed since ANN index build.")
                self._ann_metadata_warned = True
        try:
            dim = int(row["embedding_dim"])
            return dim if dim > 0 else None
        except (TypeError, ValueError):
            return None

    def _infer_embedding_dimension_sync(self, connection: sqlite3.Connection) -> int | None:
        row = connection.execute(
            """
            SELECT embedding_json
            FROM memory_vectors
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        try:
            vector = [float(item) for item in json.loads(row["embedding_json"])]
        except (TypeError, ValueError):
            return None
        return len(vector) if vector else None

    def _ensure_ann_table_sync(self, connection: sqlite3.Connection, embedding_dim: int) -> bool:
        if not self._ann_enabled or not self._ann_available or embedding_dim <= 0:
            return False
        if self._ann_dimension is not None and self._ann_dimension != embedding_dim:
            if not self._ann_dimension_warned:
                logging.warning("Embedding dimension mismatch for ANN index.")
                self._ann_dimension_warned = True
            return False
        try:
            connection.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {ANN_TABLE_NAME} USING vectorlite(
                    embedding float32[{embedding_dim}],
                    hnsw(max_elements={self._ann_max_elements})
                )
                """
            )
            self._ann_table_ready = True
            self._ann_dimension = embedding_dim
            connection.execute(
                """
                INSERT INTO memory_vector_meta (
                    id, embedding_dim, embedding_model, embedding_provider, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    embedding_dim = excluded.embedding_dim,
                    embedding_model = excluded.embedding_model,
                    embedding_provider = excluded.embedding_provider,
                    updated_at = excluded.updated_at
                """,
                (
                    ANN_META_ROW_ID,
                    embedding_dim,
                    self._embedding_model,
                    self._embedding_provider,
                    _utcnow_iso(),
                ),
            )
            return True
        except Exception:
            if not self._ann_extension_warned:
                logging.warning("Failed to initialize ANN table; fallback to linear vector search.")
                self._ann_extension_warned = True
            self._ann_table_ready = False
            return False

    def _ensure_vector_rowid_sync(self, connection: sqlite3.Connection, namespace_text: str, key: str) -> int:
        connection.execute(
            """
            INSERT OR IGNORE INTO memory_vector_ids (namespace, key)
            VALUES (?, ?)
            """,
            (namespace_text, key),
        )
        row = connection.execute(
            """
            SELECT vector_rowid
            FROM memory_vector_ids
            WHERE namespace = ? AND key = ?
            """,
            (namespace_text, key),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to allocate vector rowid.")
        return int(row["vector_rowid"])

    def _backfill_ann_index_sync(self, connection: sqlite3.Connection) -> None:
        if not self._ann_table_ready or self._ann_dimension is None or self._ann_backfill_done:
            return
        rows = connection.execute(
            """
            SELECT
                mv.namespace,
                mv.key,
                mv.embedding_json,
                mvi.vector_rowid
            FROM memory_vectors AS mv
            LEFT JOIN memory_vector_ids AS mvi
                ON mvi.namespace = mv.namespace
                AND mvi.key = mv.key
            """
        ).fetchall()
        for row in rows:
            try:
                vector = [float(item) for item in json.loads(row["embedding_json"])]
            except (TypeError, ValueError):
                continue
            if len(vector) != self._ann_dimension:
                continue
            rowid = row["vector_rowid"]
            if rowid is None:
                rowid = self._ensure_vector_rowid_sync(connection, row["namespace"], row["key"])
            connection.execute(
                f"INSERT OR REPLACE INTO {ANN_TABLE_NAME} (rowid, embedding) VALUES (?, ?)",
                (int(rowid), _vector_to_blob(vector)),
            )
        self._ann_backfill_done = True

    def _initialize_ann_state_sync(self, connection: sqlite3.Connection) -> None:
        if not self._ann_enabled or not self._ann_available:
            self._ann_table_ready = False
            return
        dim = self._read_ann_dimension_from_meta_sync(connection) or self._infer_embedding_dimension_sync(connection)
        if dim is None:
            self._ann_table_ready = False
            return
        if self._ensure_ann_table_sync(connection, dim):
            self._backfill_ann_index_sync(connection)

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | bool | None = None,
        ttl: float | None = None,
    ) -> None:
        """Store or update one memory item."""
        _ = index, ttl
        await self._ensure_initialized()
        namespace_text = _namespace_to_str(namespace)
        kind = str(value.get("kind", "unknown"))
        content = str(value.get("content", ""))
        topic = str(value.get("topic", ""))
        summary = str(value.get("summary", ""))
        created_at = str(value.get("created_at") or _utcnow_iso())
        updated_at = str(value.get("updated_at") or _utcnow_iso())
        payload_json = json.dumps(value, ensure_ascii=False)
        text_for_embedding = " ".join(part for part in [content, summary, topic] if part)
        embedding: list[float] | None = None
        if self._embed_text is not None and text_for_embedding:
            embedding = await self._embed_text(text_for_embedding)
        async with self._write_lock:
            await asyncio.to_thread(
                self._aput_sync,
                namespace_text,
                str(key),
                kind,
                content,
                topic,
                summary,
                payload_json,
                created_at,
                updated_at,
                embedding,
            )

    def _delete_ann_vector_sync(self, connection: sqlite3.Connection, namespace_text: str, key: str) -> None:
        row = connection.execute(
            """
            SELECT vector_rowid
            FROM memory_vector_ids
            WHERE namespace = ? AND key = ?
            """,
            (namespace_text, key),
        ).fetchone()
        if row is not None and self._ann_table_ready:
            connection.execute(f"DELETE FROM {ANN_TABLE_NAME} WHERE rowid = ?", (int(row["vector_rowid"]),))
        connection.execute(
            """
            DELETE FROM memory_vector_ids
            WHERE namespace = ? AND key = ?
            """,
            (namespace_text, key),
        )

    def _aput_sync(
        self,
        namespace_text: str,
        key: str,
        kind: str,
        content: str,
        topic: str,
        summary: str,
        payload_json: str,
        created_at: str,
        updated_at: str,
        embedding: list[float] | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_items (
                    namespace, key, kind, content, topic, summary,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    kind = excluded.kind,
                    content = excluded.content,
                    topic = excluded.topic,
                    summary = excluded.summary,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (namespace_text, key, kind, content, topic, summary, payload_json, created_at, updated_at),
            )
            if self._fts_enabled:
                connection.execute("DELETE FROM memory_fts WHERE namespace = ? AND key = ?", (namespace_text, key))
                connection.execute(
                    """
                    INSERT INTO memory_fts (namespace, key, kind, content, topic, summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (namespace_text, key, kind, content, topic, summary),
                )
            if embedding is not None:
                connection.execute(
                    """
                    INSERT INTO memory_vectors (namespace, key, embedding_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        embedding_json = excluded.embedding_json,
                        updated_at = excluded.updated_at
                    """,
                    (namespace_text, key, json.dumps(embedding), updated_at),
                )
                if self._ann_enabled and self._ann_available and self._ensure_ann_table_sync(connection, len(embedding)):
                    rowid = self._ensure_vector_rowid_sync(connection, namespace_text, key)
                    connection.execute(
                        f"INSERT OR REPLACE INTO {ANN_TABLE_NAME} (rowid, embedding) VALUES (?, ?)",
                        (rowid, _vector_to_blob(embedding)),
                    )
            else:
                connection.execute("DELETE FROM memory_vectors WHERE namespace = ? AND key = ?", (namespace_text, key))
                self._delete_ann_vector_sync(connection, namespace_text, key)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete one memory item by namespace and key."""
        await self._ensure_initialized()
        namespace_text = _namespace_to_str(namespace)
        async with self._write_lock:
            await asyncio.to_thread(self._adelete_sync, namespace_text, str(key))

    def _adelete_sync(self, namespace_text: str, key: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_items WHERE namespace = ? AND key = ?", (namespace_text, key))
            connection.execute("DELETE FROM memory_vectors WHERE namespace = ? AND key = ?", (namespace_text, key))
            self._delete_ann_vector_sync(connection, namespace_text, key)
            if self._fts_enabled:
                connection.execute("DELETE FROM memory_fts WHERE namespace = ? AND key = ?", (namespace_text, key))

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Read one memory item by namespace and key."""
        _ = refresh_ttl
        await self._ensure_initialized()
        namespace_text = _namespace_to_str(namespace)
        row = await asyncio.to_thread(self._aget_sync, namespace_text, str(key))
        if row is None:
            return None
        return Item(
            value=json.loads(row["payload_json"]),
            key=row["key"],
            namespace=_namespace_from_str(row["namespace"]),
            created_at=_parse_iso_datetime(row["created_at"]),
            updated_at=_parse_iso_datetime(row["updated_at"]),
        )

    def _aget_sync(self, namespace_text: str, key: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT namespace, key, payload_json, created_at, updated_at
                FROM memory_items
                WHERE namespace = ? AND key = ?
                """,
                (namespace_text, key),
            ).fetchone()

    def _search_ann_rows_sync(
        self,
        namespace_prefix_text: str,
        query_vector: list[float],
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        if not self._ann_table_ready or self._ann_dimension is None or len(query_vector) != self._ann_dimension:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    mi.namespace,
                    mi.key,
                    mi.payload_json,
                    mi.created_at,
                    mi.updated_at,
                    ann.distance AS ann_distance
                FROM {ANN_TABLE_NAME} AS ann
                JOIN memory_vector_ids AS mvi
                    ON mvi.vector_rowid = ann.rowid
                JOIN memory_items AS mi
                    ON mi.namespace = mvi.namespace
                    AND mi.key = mvi.key
                WHERE knn_search(embedding, knn_param(?, ?))
                  AND ann.rowid IN (
                    SELECT vector_rowid
                    FROM memory_vector_ids
                    WHERE namespace = ?
                       OR namespace LIKE ?
                  )
                LIMIT ?
                """,
                (
                    _vector_to_blob(query_vector),
                    candidate_limit,
                    namespace_prefix_text,
                    f"{namespace_prefix_text}.%",
                    candidate_limit,
                ),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            distance = float(row["ann_distance"]) if row["ann_distance"] is not None else 1.0
            results.append(
                {
                    "namespace": row["namespace"],
                    "key": row["key"],
                    "payload_json": row["payload_json"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "vector_rank_score": 1.0 / (1.0 + max(0.0, distance)),
                }
            )
        results.sort(key=lambda item: item["vector_rank_score"], reverse=True)
        return results

    def _search_linear_rows_sync(
        self,
        namespace_prefix_text: str,
        query_vector: list[float],
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    mi.namespace,
                    mi.key,
                    mi.payload_json,
                    mi.created_at,
                    mi.updated_at,
                    mv.embedding_json
                FROM memory_items AS mi
                JOIN memory_vectors AS mv
                    ON mv.namespace = mi.namespace
                    AND mv.key = mi.key
                WHERE (
                    mi.namespace = ?
                    OR mi.namespace LIKE ?
                )
                ORDER BY mi.updated_at DESC
                LIMIT ?
                """,
                (
                    namespace_prefix_text,
                    f"{namespace_prefix_text}.%",
                    max(candidate_limit * 4, candidate_limit),
                ),
            ).fetchall()
        scored: list[dict[str, Any]] = []
        for row in rows:
            try:
                vector = [float(item) for item in json.loads(row["embedding_json"])]
            except (TypeError, ValueError):
                continue
            similarity = _cosine_similarity(query_vector, vector)
            if similarity <= 0.0:
                continue
            scored.append(
                {
                    "namespace": row["namespace"],
                    "key": row["key"],
                    "payload_json": row["payload_json"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "vector_rank_score": similarity,
                }
            )
        scored.sort(key=lambda item: item["vector_rank_score"], reverse=True)
        return scored[:candidate_limit]

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Search memory by namespace prefix and optional query text."""
        _ = refresh_ttl
        await self._ensure_initialized()
        prefix_text = _namespace_to_str(namespace_prefix)
        cleaned_query = (query or "").strip()
        candidate_limit = max(limit + offset, self._search_candidates)

        if not cleaned_query:
            rows = await asyncio.to_thread(self._fetch_recent_rows_sync, prefix_text, candidate_limit)
            items: list[SearchItem] = []
            for row in rows:
                value = json.loads(row["payload_json"])
                if not _value_matches_filter(value, filter):
                    continue
                items.append(
                    SearchItem(
                        namespace=_namespace_from_str(row["namespace"]),
                        key=row["key"],
                        value=value,
                        created_at=_parse_iso_datetime(row["created_at"]),
                        updated_at=_parse_iso_datetime(row["updated_at"]),
                        score=None,
                    )
                )
            return items[offset : offset + limit]

        lexical_rows = await asyncio.to_thread(
            self._fetch_query_rows_sync,
            prefix_text,
            cleaned_query,
            candidate_limit,
        )
        records: dict[str, dict[str, Any]] = {}
        for row in lexical_rows:
            record_id = _item_id(row["namespace"], row["key"])
            records[record_id] = row
        lexical_order: list[str] = []
        if lexical_rows and lexical_rows[0]["lexical_score"] is not None:
            seen_lexical_ids: set[str] = set()
            for row in lexical_rows:
                record_id = _item_id(row["namespace"], row["key"])
                if record_id in seen_lexical_ids:
                    continue
                seen_lexical_ids.add(record_id)
                lexical_order.append(record_id)
            lexical_order = lexical_order[: self._rrf_candidate_max]
        else:
            lexical_scores: dict[str, float] = {}
            for row in lexical_rows:
                record_id = _item_id(row["namespace"], row["key"])
                lexical_scores[record_id] = max(
                    lexical_scores.get(record_id, 0.0),
                    float(row["fallback_lexical"] or 0.0),
                )
            lexical_order = sorted(
                lexical_scores.keys(),
                key=lambda item_id: (
                    lexical_scores[item_id],
                    _safe_updated_epoch(records[item_id]["updated_at"]),
                ),
                reverse=True,
            )[: self._rrf_candidate_max]

        query_vector: list[float] | None = None
        if self._embed_text is not None:
            query_vector = await self._embed_text(cleaned_query)

        vector_scores: dict[str, float] = {}
        vector_order: list[str] = []
        if query_vector is not None:
            vector_rows: list[dict[str, Any]] = []
            if (
                self._ann_enabled
                and self._ann_available
                and self._ann_table_ready
                and self._ann_dimension == len(query_vector)
            ):
                vector_rows = await asyncio.to_thread(
                    self._search_ann_rows_sync,
                    prefix_text,
                    query_vector,
                    max(limit + offset, self._ann_candidates),
                )
            if not vector_rows:
                vector_rows = await asyncio.to_thread(
                    self._search_linear_rows_sync,
                    prefix_text,
                    query_vector,
                    max(limit + offset, self._ann_candidates),
                )
            for row in vector_rows:
                record_id = _item_id(row["namespace"], row["key"])
                records.setdefault(record_id, row)
                vector_scores[record_id] = max(
                    vector_scores.get(record_id, 0.0),
                    float(row["vector_rank_score"]),
                )
            vector_order = sorted(
                vector_scores.keys(),
                key=lambda item_id: (
                    vector_scores[item_id],
                    _safe_updated_epoch(records[item_id]["updated_at"]),
                ),
                reverse=True,
            )[: self._rrf_candidate_max]

        filtered_ids: list[str] = []
        for item_id, row in records.items():
            value = json.loads(row["payload_json"])
            if not _value_matches_filter(value, filter):
                continue
            row["value"] = value
            filtered_ids.append(item_id)
        if not filtered_ids:
            return []

        filtered_set = set(filtered_ids)
        lexical_ranked = [item_id for item_id in lexical_order if item_id in filtered_set]
        vector_ranked = [item_id for item_id in vector_order if item_id in filtered_set]

        fused = reciprocal_rank_fusion([lexical_ranked, vector_ranked], self._rrf_k)
        lexical_rank_position = {item_id: rank for rank, item_id in enumerate(lexical_ranked, start=1)}
        vector_rank_position = {item_id: rank for rank, item_id in enumerate(vector_ranked, start=1)}
        fallback_rank = 10**9
        sorted_ids = sorted(
            filtered_ids,
            key=lambda item_id: (
                -fused.get(item_id, 0.0),
                lexical_rank_position.get(item_id, fallback_rank),
                vector_rank_position.get(item_id, fallback_rank),
                -_safe_updated_epoch(records[item_id]["updated_at"]),
            ),
        )

        sliced = sorted_ids[offset : offset + limit]
        return [
            SearchItem(
                namespace=_namespace_from_str(records[item_id]["namespace"]),
                key=records[item_id]["key"],
                value=records[item_id]["value"],
                created_at=_parse_iso_datetime(records[item_id]["created_at"]),
                updated_at=_parse_iso_datetime(records[item_id]["updated_at"]),
                score=fused.get(item_id, 0.0),
            )
            for item_id in sliced
        ]

    def _fetch_query_rows_sync(
        self,
        namespace_prefix_text: str,
        query: str,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows: list[sqlite3.Row] = []
            if self._fts_enabled:
                try:
                    rows = connection.execute(
                        """
                        SELECT
                            mi.namespace,
                            mi.key,
                            mi.payload_json,
                            mi.created_at,
                            mi.updated_at,
                            bm25(memory_fts) AS lexical_score
                        FROM memory_fts
                        JOIN memory_items AS mi
                            ON mi.namespace = memory_fts.namespace
                            AND mi.key = memory_fts.key
                        WHERE memory_fts MATCH ?
                            AND (
                                mi.namespace = ?
                                OR mi.namespace LIKE ?
                            )
                        ORDER BY lexical_score ASC, mi.updated_at DESC
                        LIMIT ?
                        """,
                        (
                            query,
                            namespace_prefix_text,
                            f"{namespace_prefix_text}.%",
                            candidate_limit,
                        ),
                    ).fetchall()
                except sqlite3.OperationalError:
                    if not self._warned_fts_query:
                        logging.warning(
                            "FTS5 query parsing failed for '%s'. Falling back to LIKE search.",
                            query,
                        )
                        self._warned_fts_query = True
                    rows = []
            if not rows:
                rows = connection.execute(
                    """
                    SELECT
                        mi.namespace,
                        mi.key,
                        mi.payload_json,
                        mi.created_at,
                        mi.updated_at,
                        NULL AS lexical_score
                    FROM memory_items AS mi
                    WHERE (
                        mi.namespace = ?
                        OR mi.namespace LIKE ?
                    )
                    AND (
                        LOWER(COALESCE(mi.content, '')) LIKE ?
                        OR LOWER(COALESCE(mi.topic, '')) LIKE ?
                        OR LOWER(COALESCE(mi.summary, '')) LIKE ?
                    )
                    ORDER BY mi.updated_at DESC
                    LIMIT ?
                    """,
                    (
                        namespace_prefix_text,
                        f"{namespace_prefix_text}.%",
                        f"%{query.lower()}%",
                        f"%{query.lower()}%",
                        f"%{query.lower()}%",
                        candidate_limit,
                    ),
                ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            text_blob = " ".join(str(payload.get(field, "")) for field in ["content", "topic", "summary"])
            results.append(
                {
                    "namespace": row["namespace"],
                    "key": row["key"],
                    "payload_json": row["payload_json"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "lexical_score": row["lexical_score"],
                    "fallback_lexical": _fallback_lexical_score(query, text_blob),
                }
            )
        return results

    def _fetch_recent_rows_sync(
        self,
        namespace_prefix_text: str,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    mi.namespace,
                    mi.key,
                    mi.payload_json,
                    mi.created_at,
                    mi.updated_at
                FROM memory_items AS mi
                WHERE (
                    mi.namespace = ?
                    OR mi.namespace LIKE ?
                )
                ORDER BY mi.updated_at DESC
                LIMIT ?
                """,
                (
                    namespace_prefix_text,
                    f"{namespace_prefix_text}.%",
                    candidate_limit,
                ),
            ).fetchall()
        return [
            {
                "namespace": row["namespace"],
                "key": row["key"],
                "payload_json": row["payload_json"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]


_SQLITE_STORE_CACHE: dict[str, SQLiteMemoryStore] = {}


def get_sqlite_memory_store(configurable: Configuration) -> SQLiteMemoryStore:
    """Get cached SQLite store instance for configured DB path."""
    path = str(Path(configurable.memory_sqlite_path).resolve())
    store = _SQLITE_STORE_CACHE.get(path)
    embed_text = build_embedding_function(configurable)
    if store is None:
        store = SQLiteMemoryStore(
            db_path=path,
            journal_mode=configurable.memory_sqlite_journal_mode,
            busy_timeout_ms=configurable.memory_sqlite_busy_timeout_ms,
            search_candidates=configurable.memory_search_candidates,
            ann_enabled=configurable.memory_ann_enabled,
            ann_max_elements=configurable.memory_ann_max_elements,
            ann_candidates=configurable.memory_ann_candidates,
            rrf_k=configurable.memory_rrf_k,
            rrf_candidate_max=configurable.memory_rrf_candidate_max,
            vector_weight=configurable.memory_vector_weight,
            embed_text=embed_text,
        )
        _SQLITE_STORE_CACHE[path] = store
    store.update_runtime_settings(configurable, embed_text=embed_text)
    return store
