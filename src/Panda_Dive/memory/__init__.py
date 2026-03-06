"""Public memory orchestration API for Panda_Dive."""

import logging

from langchain_core.runnables import RunnableConfig

from ..configuration import Configuration
from .extractor import MemoryMutationBatch, extract_memory_mutations
from .injector import build_memory_injection_block
from .retriever import retrieve_memory_bundle
from .schemas import MemoryBundle
from .store import delete_memory, resolve_namespace, upsert_episode, upsert_fact


async def _apply_memory_mutations(
    mutations: MemoryMutationBatch,
    config: RunnableConfig,
    namespace: tuple[str, ...],
) -> None:
    """Apply LangMem-produced memory mutations."""
    for memory_id in mutations.deletes:
        await delete_memory(memory_id, config, namespace=namespace)
    for fact in mutations.facts:
        await upsert_fact(fact, config, namespace=namespace)
    for episode in mutations.episodes:
        await upsert_episode(episode, config, namespace=namespace)


async def _reconcile_and_persist(
    *,
    topic: str,
    compressed_research: str,
    raw_notes: str,
    config: RunnableConfig,
    source_run_id: str,
    source_message_ids: list[str] | None = None,
    namespace: tuple[str, ...],
) -> None:
    """Extract, reconcile, and persist LangMem-backed research memories."""
    mutations = await extract_memory_mutations(
        topic=topic,
        text=compressed_research,
        raw_notes=raw_notes,
        config=config,
        source_run_id=source_run_id,
        source_message_ids=source_message_ids,
        namespace=namespace,
    )
    await _apply_memory_mutations(mutations, config, namespace)


async def persist_research_memory(
    *,
    topic: str,
    compressed_research: str,
    raw_notes: str,
    config: RunnableConfig,
    source_run_id: str,
    source_message_ids: list[str] | None = None,
) -> None:
    """Persist memory from a researcher compression output."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_enabled:
        return

    namespace = resolve_namespace(config, topic=topic)
    await _reconcile_and_persist(
        topic=topic,
        compressed_research=compressed_research,
        raw_notes=raw_notes,
        config=config,
        source_run_id=source_run_id,
        source_message_ids=source_message_ids,
        namespace=namespace,
    )


async def persist_final_report_memory(
    *,
    report_text: str,
    notes: list[str],
    topic: str,
    config: RunnableConfig,
    source_run_id: str,
) -> None:
    """Persist memory extracted from final report."""
    try:
        await persist_research_memory(
            topic=topic,
            compressed_research=report_text,
            raw_notes="\n".join(notes),
            config=config,
            source_run_id=source_run_id,
        )
    except Exception:
        logging.exception("Failed to persist final report memory")


async def retrieve_memory_for_prompt(
    *,
    query: str,
    task_context: str,
    config: RunnableConfig,
    topic: str | None = None,
) -> tuple[MemoryBundle, str]:
    """Retrieve memory bundle and render block for system prompt."""
    bundle = await retrieve_memory_bundle(
        query=query,
        task_context=task_context,
        config=config,
        topic=topic,
    )
    configurable = Configuration.from_runnable_config(config)
    block = build_memory_injection_block(
        facts=bundle.facts,
        episodes=bundle.episodes,
        preferences=bundle.preferences,
        max_tokens=configurable.memory_max_injection_tokens,
    )
    return bundle, block


__all__ = [
    "build_memory_injection_block",
    "persist_final_report_memory",
    "persist_research_memory",
    "retrieve_memory_for_prompt",
]
