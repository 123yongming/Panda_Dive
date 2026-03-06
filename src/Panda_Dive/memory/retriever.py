"""Memory retrieval logic."""

from langchain_core.runnables import RunnableConfig

from ..configuration import Configuration
from .schemas import MemoryBundle
from .store import search_episodes, search_facts


async def retrieve_memory_bundle(
    query: str,
    task_context: str,
    config: RunnableConfig,
    *,
    topic: str | None = None,
) -> MemoryBundle:
    """Retrieve memory bundle for prompt injection."""
    _ = task_context
    configurable = Configuration.from_runnable_config(config)
    if not configurable.memory_enabled:
        return MemoryBundle()

    facts = await search_facts(
        query,
        config,
        top_k=configurable.memory_retrieval_top_k,
        topic=topic,
    )
    episodes = await search_episodes(
        query,
        config,
        top_k=5,
        topic=topic,
    )
    preferences = [item.content for item in facts if item.fact_type == "preference"][:5]
    return MemoryBundle(
        facts=facts,
        episodes=episodes,
        preferences=preferences,
    )

