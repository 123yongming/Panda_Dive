"""LangMem-powered memory extraction and reconciliation utilities."""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..configuration import Configuration
from ..utils import get_api_key_for_model, get_init_chat_model_params
from .schemas import MemoryEpisode, MemoryFact
from .store import resolve_namespace, search_episodes, search_facts

URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")

LANGMEM_INSTRUCTIONS = """You manage long-term research memory for Panda_Dive.

You will receive a completed research topic, the compressed research result, and supporting notes.
Only produce two kinds of memories:

1. ResearchFactMemory
   - Durable, reusable facts that should inform future work on the topic
   - Each fact must be a single standalone statement
   - Update or delete existing facts instead of duplicating them

2. ResearchEpisodeMemory
   - One concise episodic snapshot for this completed research run
   - Summarize the run at a high level with the most important findings
   - Update or delete existing episodic summaries if the new run supersedes them

Rules:
- Stay grounded in the provided research summary and supporting notes
- Prefer dense, precise wording over verbose prose
- Do not include citations, ids, timestamps, or storage metadata in generated fields
- Merge or prune redundant memories instead of keeping overlapping copies
- If a memory is contradicted or obsolete, update it or delete it
- Only create memories that are useful in future research prompts
"""

MAX_EXISTING_FACTS = 64
MAX_EXISTING_EPISODES = 16


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text in first-seen order."""
    if not text:
        return []
    urls: list[str] = []
    for url in URL_PATTERN.findall(text):
        if url not in urls:
            urls.append(url)
    return urls


def _merge_unique(existing: list[str], new: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*existing, *new]:
        if value and value not in merged:
            merged.append(value)
    return merged


def _clamp_unit(value: float | None, *, default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def _normalize_findings(findings: list[str], summary: str) -> list[str]:
    cleaned = [item.strip() for item in findings if item and item.strip()]
    if cleaned:
        return cleaned[:5]
    return [line.strip("- ").strip() for line in summary.splitlines() if line.strip()][:5]


def _get_memory_manager_factory():
    try:
        from langmem import create_memory_manager
    except ImportError as exc:
        raise RuntimeError(
            "langmem is required for memory extraction and namespace isolation. "
            "Install project dependencies to enable memory."
        ) from exc
    return create_memory_manager


class ResearchFactMemory(BaseModel):
    """Structured durable fact extracted by LangMem."""

    fact_type: str = Field(
        default="knowledge",
        description="Fact subtype. Use 'knowledge' for research facts.",
    )
    content: str = Field(
        description="A single standalone fact worth reusing in future research prompts.",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in the fact after reconciling with existing memories.",
    )
    status: str = Field(
        default="active",
        description="Use 'active' for normal facts or 'disputed' when the fact remains contested.",
    )


class ResearchEpisodeMemory(BaseModel):
    """Structured episodic summary extracted by LangMem."""

    topic: str = Field(description="Topic for the completed research run.")
    summary: str = Field(
        description="A concise episodic summary of the completed research run.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Short bullet findings distilled from the run.",
    )
    quality_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the quality and usefulness of the episodic summary.",
    )


@dataclass
class MemoryMutationBatch:
    """Persistent memory mutations produced by LangMem reconciliation."""

    facts: list[MemoryFact] = field(default_factory=list)
    episodes: list[MemoryEpisode] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)


_MANAGER_CACHE: dict[tuple[str, int, str], Any] = {}


def _build_langmem_chat_model(config: RunnableConfig):
    """Build the base chat model expected by LangMem."""
    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.compression_model
    api_key = get_api_key_for_model(model_name, config) or ""
    model_params = get_init_chat_model_params(model_name)
    return init_chat_model(
        model=model_name,
        max_tokens=configurable.compression_model_max_tokens,
        api_key=api_key,
        **model_params,
    )


def get_memory_manager(config: RunnableConfig):
    """Build or reuse a cached LangMem manager."""
    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.compression_model
    max_tokens = configurable.compression_model_max_tokens
    api_key = get_api_key_for_model(model_name, config) or ""
    cache_key = (model_name, max_tokens, api_key)
    manager = _MANAGER_CACHE.get(cache_key)
    if manager is not None:
        return manager

    model = _build_langmem_chat_model(config)
    manager = _get_memory_manager_factory()(
        model,
        schemas=[ResearchFactMemory, ResearchEpisodeMemory],
        instructions=LANGMEM_INSTRUCTIONS,
        enable_inserts=True,
        enable_updates=True,
        enable_deletes=True,
    )
    _MANAGER_CACHE[cache_key] = manager
    return manager


def _build_session_messages(
    *,
    topic: str,
    text: str,
    raw_notes: str,
) -> list[HumanMessage]:
    sections: list[str] = []
    if topic.strip():
        sections.append(f"<research_topic>\n{topic.strip()}\n</research_topic>")
    if text.strip():
        sections.append(f"<research_summary>\n{text.strip()}\n</research_summary>")
    if raw_notes.strip():
        sections.append(f"<supporting_notes>\n{raw_notes.strip()}\n</supporting_notes>")
    if not sections:
        return []
    return [HumanMessage(content="\n\n".join(sections))]


def _to_existing_payload(
    facts: list[MemoryFact],
    episodes: list[MemoryEpisode],
) -> list[tuple[str, str, BaseModel]]:
    existing: list[tuple[str, str, BaseModel]] = []
    for fact in facts:
        existing.append(
            (
                fact.id,
                ResearchFactMemory.__name__,
                ResearchFactMemory(
                    fact_type=fact.fact_type,
                    content=fact.content,
                    confidence=fact.confidence,
                    status=fact.status,
                ),
            )
        )
    for episode in episodes:
        existing.append(
            (
                episode.id,
                ResearchEpisodeMemory.__name__,
                ResearchEpisodeMemory(
                    topic=episode.topic,
                    summary=episode.summary,
                    key_findings=list(episode.key_findings),
                    quality_score=episode.quality_score,
                ),
            )
        )
    return existing


def _is_remove_doc(value: Any) -> bool:
    return hasattr(value, "__repr_name__") and value.__repr_name__() == "RemoveDoc"


def _memory_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _fact_signature(fact: MemoryFact) -> dict[str, Any]:
    payload = fact.model_dump()
    payload.pop("updated_at", None)
    payload.pop("rank_score", None)
    return payload


def _episode_signature(episode: MemoryEpisode) -> dict[str, Any]:
    payload = episode.model_dump()
    payload.pop("rank_score", None)
    return payload


def _build_fact(
    *,
    memory_id: str | None,
    namespace: str,
    content: Any,
    existing: MemoryFact | None,
    source_run_id: str,
    source_message_ids: list[str],
    source_urls: list[str],
) -> MemoryFact:
    merged_urls = _merge_unique(existing.source_urls if existing else [], source_urls)
    merged_message_ids = _merge_unique(
        existing.source_message_ids if existing else [],
        source_message_ids,
    )
    return MemoryFact(
        id=memory_id or (existing.id if existing else _memory_id("fact")),
        namespace=namespace,
        fact_type=str(getattr(content, "fact_type", None) or (existing.fact_type if existing else "knowledge")),
        content=str(getattr(content, "content", "")).strip(),
        confidence=_clamp_unit(
            getattr(content, "confidence", None),
            default=existing.confidence if existing else 0.85,
        ),
        novelty=existing.novelty if existing else 1.0,
        source_urls=merged_urls,
        source_run_id=source_run_id,
        source_message_ids=merged_message_ids,
        created_at=existing.created_at if existing else _utcnow_iso(),
        updated_at=_utcnow_iso(),
        ttl_days=existing.ttl_days if existing else None,
        status=str(getattr(content, "status", None) or (existing.status if existing else "active")),
        rank_score=existing.rank_score if existing else 0.0,
    )


def _build_episode(
    *,
    memory_id: str | None,
    namespace: str,
    fallback_topic: str,
    content: Any,
    existing: MemoryEpisode | None,
    citations: list[str],
) -> MemoryEpisode:
    merged_citations = _merge_unique(existing.citations if existing else [], citations)
    summary = str(getattr(content, "summary", "")).strip()
    return MemoryEpisode(
        id=memory_id or (existing.id if existing else _memory_id("episode")),
        namespace=namespace,
        topic=str(getattr(content, "topic", None) or (existing.topic if existing else fallback_topic)),
        summary=summary,
        key_findings=_normalize_findings(
            list(getattr(content, "key_findings", []) or []),
            summary,
        ),
        citations=merged_citations,
        quality_score=_clamp_unit(
            getattr(content, "quality_score", None),
            default=existing.quality_score if existing else 0.8,
        ),
        created_at=existing.created_at if existing else _utcnow_iso(),
        rank_score=existing.rank_score if existing else 0.0,
    )


async def extract_memory_mutations(
    *,
    topic: str,
    text: str,
    raw_notes: str,
    config: RunnableConfig,
    source_run_id: str,
    source_message_ids: list[str] | None = None,
    namespace: tuple[str, ...] | None = None,
) -> MemoryMutationBatch:
    """Reconcile stored memories against a new completed research result."""
    summary_text = (text or "").strip()
    if not summary_text:
        return MemoryMutationBatch()

    effective_namespace = namespace or resolve_namespace(config, topic=topic)
    namespace_str = ".".join(effective_namespace)
    source_urls = extract_urls(raw_notes)
    message_ids = source_message_ids or []

    configurable = Configuration.from_runnable_config(config)
    existing_facts = await search_facts(
        None,
        config,
        top_k=min(MAX_EXISTING_FACTS, configurable.memory_max_facts_per_namespace),
        topic=topic,
        namespace=effective_namespace,
    )
    existing_episodes = await search_episodes(
        None,
        config,
        top_k=MAX_EXISTING_EPISODES,
        topic=topic,
        namespace=effective_namespace,
    )
    facts_by_id = {fact.id: fact for fact in existing_facts}
    episodes_by_id = {episode.id: episode for episode in existing_episodes}

    manager = get_memory_manager(config)
    results = await manager.ainvoke(
        {
            "messages": _build_session_messages(
                topic=topic,
                text=summary_text,
                raw_notes=raw_notes,
            ),
            "existing": _to_existing_payload(existing_facts, existing_episodes),
            "max_steps": 1,
        },
        config=config,
    )

    batch = MemoryMutationBatch()
    for extracted in results:
        memory_id = getattr(extracted, "id", None)
        content = getattr(extracted, "content", None)
        if memory_id is None and isinstance(extracted, (tuple, list)) and len(extracted) >= 2:
            memory_id = extracted[0]
            content = extracted[1]
        if memory_id is None or content is None:
            continue
        if _is_remove_doc(content):
            if memory_id not in batch.deletes:
                batch.deletes.append(memory_id)
            continue
        if isinstance(content, ResearchFactMemory):
            existing_fact = facts_by_id.get(memory_id)
            fact = _build_fact(
                memory_id=memory_id if existing_fact else None,
                namespace=namespace_str,
                content=content,
                existing=existing_fact,
                source_run_id=source_run_id,
                source_message_ids=message_ids,
                source_urls=source_urls,
            )
            if existing_fact and _fact_signature(fact) == _fact_signature(existing_fact):
                continue
            batch.facts.append(fact)
            continue
        if isinstance(content, ResearchEpisodeMemory):
            existing_episode = episodes_by_id.get(memory_id)
            episode = _build_episode(
                memory_id=memory_id if existing_episode else None,
                namespace=namespace_str,
                fallback_topic=topic,
                content=content,
                existing=existing_episode,
                citations=source_urls,
            )
            if existing_episode and _episode_signature(episode) == _episode_signature(existing_episode):
                continue
            batch.episodes.append(episode)
    return batch
