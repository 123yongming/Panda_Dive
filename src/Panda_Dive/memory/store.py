"""Storage helpers for Panda_Dive memory facts and episodes."""

import hashlib
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.store.base import BaseStore

from ..configuration import Configuration
from .schemas import MemoryEpisode, MemoryFact
from .sqlite_backend import get_sqlite_memory_store


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _owner_from_config(config: RunnableConfig) -> str:
    configurable = config.get("configurable", {}) if config else {}
    metadata = config.get("metadata", {}) if config else {}
    return str(
        configurable.get("owner")
        or metadata.get("owner")
        or configurable.get("thread_id")
        or "default"
    )


def _thread_from_config(config: RunnableConfig) -> str:
    configurable = config.get("configurable", {}) if config else {}
    return str(configurable.get("thread_id") or "thread")


def _topic_key(topic: str | None) -> str:
    if not topic:
        return "topic"
    return hashlib.sha1(topic.encode("utf-8")).hexdigest()[:12]


def _get_namespace_template_cls():
    try:
        from langmem.utils import NamespaceTemplate
    except ImportError as exc:
        raise RuntimeError(
            "langmem is required for memory extraction and namespace isolation. "
            "Install project dependencies to enable memory."
        ) from exc
    return NamespaceTemplate


def _namespace_segments(template: str) -> tuple[str, ...]:
    segments = tuple(segment for segment in str(template or "").split(".") if segment)
    if segments:
        return segments
    return ("memory", "owner", "default")


def _namespace_runtime_config(
    config: RunnableConfig,
    *,
    topic: str | None = None,
) -> RunnableConfig:
    runtime_config = dict(config or {})
    configurable = dict(runtime_config.get("configurable", {}))
    configurable["owner"] = _owner_from_config(config)
    configurable["thread_id"] = _thread_from_config(config)
    configurable["topic_hash"] = _topic_key(topic)
    runtime_config["configurable"] = configurable
    return runtime_config


def resolve_namespace(config: RunnableConfig, topic: str | None = None) -> tuple[str, ...]:
    """Resolve memory namespace from runtime config."""
    configurable = Configuration.from_runnable_config(config)
    namespace_template = _get_namespace_template_cls()(
        _namespace_segments(configurable.memory_namespace_template)
    )
    resolved = namespace_template(_namespace_runtime_config(config, topic=topic))
    return tuple(str(part) for part in resolved)


def _namespace_str(namespace: tuple[str, ...]) -> str:
    return ".".join(namespace)


def _namespace_from_str(namespace: str) -> tuple[str, ...]:
    if not namespace:
        return tuple()
    return tuple(part for part in namespace.split(".") if part)


def _store_from_arg(
    config: RunnableConfig,
    store: BaseStore | None,
) -> Any:
    if store is not None:
        return store
    configurable = Configuration.from_runnable_config(config)
    if configurable.memory_backend == "sqlite":
        return get_sqlite_memory_store(configurable)
    return get_store()


def _as_memory_fact(payload: dict[str, Any], score: float | None = None) -> MemoryFact:
    value = dict(payload)
    value.pop("kind", None)
    value.setdefault("rank_score", score or 0.0)
    return MemoryFact(**value)


def _as_memory_episode(payload: dict[str, Any], score: float | None = None) -> MemoryEpisode:
    value = dict(payload)
    value.pop("kind", None)
    value.setdefault("rank_score", score or 0.0)
    return MemoryEpisode(**value)


async def _trim_facts(namespace: tuple[str, ...], configurable: Configuration, store: BaseStore) -> None:
    items = await store.asearch(
        namespace,
        filter={"kind": "fact"},
        limit=max(configurable.memory_max_facts_per_namespace, 1) + 256,
    )
    if len(items) <= configurable.memory_max_facts_per_namespace:
        return
    ordered = sorted(
        items,
        key=lambda item: (
            float(item.value.get("confidence", 0.0)),
            item.value.get("updated_at", ""),
        ),
        reverse=True,
    )
    keep_keys = {
        item.key for item in ordered[: configurable.memory_max_facts_per_namespace]
    }
    for item in items:
        if item.key not in keep_keys:
            await store.adelete(item.namespace, item.key)


async def upsert_fact(
    fact: MemoryFact,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    namespace: tuple[str, ...] | None = None,
) -> bool:
    """Upsert a memory fact without heuristic governance."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_enabled:
        return False

    store_client = _store_from_arg(config, store)
    effective_namespace = (
        namespace
        or _namespace_from_str(fact.namespace)
        or resolve_namespace(config)
    )
    fact.namespace = _namespace_str(effective_namespace)
    if not fact.updated_at:
        fact.updated_at = _utcnow_iso()
    payload = fact.model_dump()
    payload["kind"] = "fact"
    await store_client.aput(effective_namespace, fact.id, payload, index=["content"])
    await _trim_facts(effective_namespace, configurable, store_client)
    return True


async def upsert_episode(
    episode: MemoryEpisode,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    namespace: tuple[str, ...] | None = None,
) -> bool:
    """Upsert an episodic memory entry."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_enabled:
        return False
    effective_namespace = (
        namespace
        or _namespace_from_str(episode.namespace)
        or resolve_namespace(config, topic=episode.topic)
    )
    episode.namespace = _namespace_str(effective_namespace)
    payload = episode.model_dump()
    payload["kind"] = "episode"
    store_client = _store_from_arg(config, store)
    await store_client.aput(
        effective_namespace,
        episode.id,
        payload,
        index=["summary", "topic"],
    )
    return True


async def search_facts(
    query: str | None,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    top_k: int | None = None,
    topic: str | None = None,
    namespace: tuple[str, ...] | None = None,
) -> list[MemoryFact]:
    """Search stored memory facts."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_enabled:
        return []
    limit = top_k or configurable.memory_retrieval_top_k
    effective_namespace = namespace or resolve_namespace(config, topic=topic)
    store_client = _store_from_arg(config, store)
    items = await store_client.asearch(
        effective_namespace,
        query=query,
        filter={"kind": "fact"},
        limit=limit,
    )
    return [_as_memory_fact(item.value, score=item.score or 0.0) for item in items]


async def search_episodes(
    query: str | None,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    top_k: int = 5,
    topic: str | None = None,
    namespace: tuple[str, ...] | None = None,
) -> list[MemoryEpisode]:
    """Search episodic memories."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_enabled:
        return []
    effective_namespace = namespace or resolve_namespace(config, topic=topic)
    store_client = _store_from_arg(config, store)
    items = await store_client.asearch(
        effective_namespace,
        query=query,
        filter={"kind": "episode"},
        limit=top_k,
    )
    return [_as_memory_episode(item.value, score=item.score or 0.0) for item in items]


async def delete_memory(
    memory_id: str,
    config: RunnableConfig,
    *,
    store: BaseStore | None = None,
    namespace: tuple[str, ...] | None = None,
    topic: str | None = None,
) -> None:
    """Delete one memory item by namespace and id."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_enabled:
        return
    effective_namespace = namespace or resolve_namespace(config, topic=topic)
    store_client = _store_from_arg(config, store)
    await store_client.adelete(effective_namespace, memory_id)
