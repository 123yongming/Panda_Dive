# Panda_Dive Agent Guidelines

Guidelines for agents working on this deep domain research tool built with LangGraph and LangChain.

## Build, Lint, Test Commands

```bash
# Install
pip install -e .
pip install -e ".[dev]"

# Testing (tests located in src/test_*.py)
python -m pytest                           # Run all tests
python -m pytest src/test_api.py           # Run specific test file
python -m pytest src/test_api.py::test_fn  # Run single test by function
python -m pytest -k "test_name"            # Run tests by name pattern

# Linting (ruff configured in pyproject.toml)
ruff check .                               # Check code
ruff check --fix .                         # Auto-fix issues
mypy src/Panda_Dive/                       # Type checking
```

## Code Style

### Python Standards
- **Version**: Python 3.10+
- **Type Hints**: Mandatory for all functions; use `list[str]`, `dict[str, Any]` (not `List`, `Dict`)
- **Imports**: stdlib → third-party → local; grouped with blank lines; ruff handles isort
- **Docstrings**: Google-style, imperative mood (enforced by ruff D401)

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Functions/variables | `snake_case` | `tavily_search`, `research_brief` |
| Classes | `PascalCase` | `Configuration`, `AgentState` |
| Constants | `UPPER_SNAKE_CASE` | `TAVILY_SEARCH_DESCRIPTION` |
| Private members | `_leading_underscore` | `_check_token_limit` |
| Async functions | `async def` prefix | `async def summarize_webpage` |

### Key Patterns

**Async/Await (required for all LangGraph nodes):**
```python
async def research_node(state: AgentState) -> dict:
    results = await asyncio.gather(*[agent.ainvoke(...) for agent in agents])
    return {"findings": results}
```

**LangGraph State Management:**
```python
class AgentState(MessagesState):
    messages: Annotated[list[Message], override_reducer]
    research_brief: Optional[str]
```

**Error Handling:**
```python
try:
    result = await operation()
except TokenLimitError:
    logging.exception("Token limit exceeded")
    return f"Error: {e}"
except Exception as e:
    logging.error(f"Operation failed: {e}")
    raise
```

**Pydantic Models:**
```python
class Configuration(BaseModel):
    max_iterations: int = Field(
        default=6,
        description="Max iterations",
        metadata={"x_oap_ui_config": {"type": "number", "min": 1, "max": 10}}
    )
```

**Tools:**
```python
@tool(description="Search the web")
async def web_search(query: str, max_results: int = 5) -> str:
    try:
        return format_results(await search_api(query, max_results))
    except Exception as e:
        return f"Search failed: {e}"
```

**Structured Output with Fallback:**
```python
if supports_structured_output(model_name):
    model = base_model.with_structured_output(OutputType).with_retry(stop_after_attempt=3)
else:
    model = base_model.with_retry(stop_after_attempt=3)
```

**Logging:**
- `logging.warning()` for recoverable issues (fallbacks, retries)
- `logging.error()` for critical failures
- `logging.exception()` for exceptions with stack trace

## Project Architecture

### Main Graph (`deepresearcher.py`)
```
START → clarify_with_user → write_research_brief → research_supervisor → final_report_generation → END
```

### Supervisor Subgraph
```
supervisor → supervisor_tools → (loop or complete)
```

### Researcher Subgraph
```
researcher → researcher_tools → compress_research → Done
```

### Key Files
| File | Purpose |
|------|---------|
| `deepresearcher.py` | Main graph orchestration |
| `configuration.py` | Pydantic configuration models |
| `state.py` | TypedDict state definitions |
| `prompts.py` | System prompts for LLMs |
| `utils.py` | Tool wrappers, API keys, MCP integration |
| `retrieval_quality.py` | Query rewriting, scoring, reranking |

## Common Tasks

### Adding a Tool
1. Define in `utils.py` with `@tool()` decorator
2. Add to `get_all_tools()`
3. Update prompts if needed

### Adding Config
1. Add field to `Configuration` in `configuration.py`
2. Include `Field()` with `x_oap_ui_config` metadata for UI
3. Access via `Configuration.from_runnable_config(config)`

### State Updates
- Use `Command(goto=..., update=...)` for transitions
- Use `{"type": "override", "value": [...]}` to replace (not append)
- Default is append via `operator.add`

### Model Initialization
```python
model = create_chat_model(
    model_name=configurable.research_model,
    max_tokens=configurable.research_model_max_tokens,
    api_key=get_api_key_for_model(configurable.research_model, config),
    tags=["langsmith:nostream"],
)
```

## Environment Variables

Required for development:
- `TAVILY_API_KEY` - For Tavily search
- `ARK_API_KEY` - For VolcEngine models
- `DEEPSEEK_API_KEY` - For DeepSeek models
- `GET_API_KEYS_FROM_CONFIG` - Set "true" for production

See `.env.example` for all options.
