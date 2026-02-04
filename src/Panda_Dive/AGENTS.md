# Panda_Dive Package

Deep research multi-agent system - LangGraph orchestration with supervisor/researcher subgraphs.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Main graph entry | `deepresearcher.py` | Compiled graph at bottom, supervisor & researcher subgraphs |
| Runtime config | `configuration.py` | `Configuration` model, `SearchAPI` enum, validation |
| State definitions | `state.py` | TypedDict states with `Annotated[...]` reducers |
| LLM prompts | `prompts.py` | All system prompts, imported by researcher |
| Tool wrappers | `utils.py` | `create_chat_model()`, search APIs, MCP loading |
| Retrieval quality | `retrieval_quality.py` | Query rewriting, scoring, reranking logic |
| Package exports | `__init__.py` | Exports `Configuration`, `deep_researcher` only |

## CONVENTIONS

- **Python 3.10+ types**: `list[str]`, `dict[str, Any]`, `str \| None` (no `Optional`)
- **Annotated state**: All list fields use `Annotated[list[T], override_reducer]`
- **Async nodes**: ALL graph nodes are async, return `Command(goto=..., update={...})`
- **Model initialization**: Use `create_chat_model()` from utils.py, never `init_chat_model` directly
- **Retry**: LLM calls chain `.with_retry(stop_after_attempt=3)`
- **LangSmith tags**: Internal calls use `tags=["langsmith:nostream"]`
- **Bilingual docs**: English docstrings required, Chinese comments acceptable

## ANTI-PATTERNS

- **NEVER** use `Optional[T]` - always `T \| None`
- **NEVER** call `deep_researcher.invoke()` - use `ainvoke()` (async only)
- **NEVER** modify state directly - always return `Command(update={...})`
- **NEVER** import `init_chat_model` directly - use `create_chat_model()` wrapper
- **DO NOT** add non-async nodes - breaks LangGraph async execution
- **DO NOT** use list concatenation without `override_reducer` annotation
