"""Memory prompt injection utilities."""

from .schemas import MemoryEpisode, MemoryFact


def estimate_token_count(text: str) -> int:
    """Estimate token count using character approximation."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _sort_facts(facts: list[MemoryFact]) -> list[MemoryFact]:
    return sorted(
        facts,
        key=lambda item: (item.rank_score, item.confidence, item.novelty),
        reverse=True,
    )


def _sort_episodes(episodes: list[MemoryEpisode]) -> list[MemoryEpisode]:
    return sorted(
        episodes,
        key=lambda item: (item.rank_score, item.quality_score),
        reverse=True,
    )


def _fit_lines_to_budget(lines: list[str], max_tokens: int) -> list[str]:
    fitted: list[str] = []
    for line in lines:
        candidate = "\n".join([*fitted, line]) if fitted else line
        if estimate_token_count(candidate) > max_tokens:
            break
        fitted.append(line)
    return fitted


def build_memory_injection_block(
    facts: list[MemoryFact],
    episodes: list[MemoryEpisode],
    preferences: list[str],
    max_tokens: int,
) -> str:
    """Build a budget-aware memory block for system prompt injection."""
    if max_tokens <= 0:
        return ""

    facts_budget = max(1, int(max_tokens * 0.6))
    episodes_budget = max(1, int(max_tokens * 0.3))
    preferences_budget = max(1, max_tokens - facts_budget - episodes_budget)

    fact_lines = [
        f"- [{fact.fact_type}] {fact.content} (conf={fact.confidence:.2f})"
        for fact in _sort_facts(facts)
    ]
    episode_lines = [
        f"- [{episode.topic}] {episode.summary} (q={episode.quality_score:.2f})"
        for episode in _sort_episodes(episodes)
    ]
    preference_lines = [f"- {item}" for item in preferences]

    fitted_facts = _fit_lines_to_budget(fact_lines, facts_budget)
    fitted_episodes = _fit_lines_to_budget(episode_lines, episodes_budget)
    fitted_preferences = _fit_lines_to_budget(preference_lines, preferences_budget)

    sections: list[str] = ["<memory_context>"]
    if fitted_facts:
        sections.append("Facts:")
        sections.extend(fitted_facts)
    if fitted_episodes:
        sections.append("Episodes:")
        sections.extend(fitted_episodes)
    if fitted_preferences:
        sections.append("Preferences:")
        sections.extend(fitted_preferences)
    sections.append("</memory_context>")

    output = "\n".join(sections)
    while estimate_token_count(output) > max_tokens and len(sections) > 2:
        sections.pop(-2)
        output = "\n".join(sections)
    return output
