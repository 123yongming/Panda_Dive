# Panda_Dive Agent Guidelines

This file provides guidelines for agents working on the Panda_Dive codebase - a deep domain research tool built with LangGraph and LangChain.

## Build, Lint, Test Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest src/test_api.py

# Run single test by name
python -m pytest -k "test_name"

# Run test by function reference
python -m pytest src/test_api.py::test_function_name

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest --cov=Panda_Dive
```

### Linting
```bash
# Run ruff (configured linter/formatter)
ruff check .
ruff check src/Panda_Dive/

# Auto-fix ruff issues
ruff check --fix .

# Type checking with mypy
mypy src/Panda_Dive/
```

### Installation
```bash
# Install package in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Code Style Guidelines

### Python Version
- Requires Python 3.10+
- Use modern type hints (e.g., `list[str]` not `List[str]`)

### Import Style
- Standard library imports first
- Third-party imports second
- Local imports third
- Use `isort` (ruff handles this automatically)
- Group imports by type with blank lines

```python
# Standard library
import asyncio
import logging
from datetime import datetime

# Third-party
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

# Local
from .configuration import Configuration
from .state import AgentState
```

### Type Annotations
- **MANDATORY**: All functions must have type hints
- Use `Optional[T]` for nullable types
- Use `Annotated[type, reducer]` for LangGraph state fields
- Prefer `list[str]`, `dict[str, int]` over `List[str]`, `Dict[str, int]` (Python 3.9+ syntax)
- Use `Literal[T]` for string enum alternatives

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict

def process_data(
    data: dict[str, Any],
    config: Optional[Configuration] = None
) -> list[str]:
    """Process input data and return results."""
    ...

class AgentState(TypedDict):
    messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
```

### Naming Conventions
- **Functions/variables**: `snake_case` (e.g., `tavily_search`, `research_brief`)
- **Classes**: `PascalCase` (e.g., `Configuration`, `AgentState`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `TAVILY_SEARCH_DESCRIPTION`)
- **Private members**: `_leading_underscore` (e.g., `_check_openai_token_limit`)
- **Async functions**: `async def` with descriptive names (e.g., `async def summarize_webpage`)

### Error Handling
- Use specific exception types when possible
- Log errors with context using `logging.exception()` or `logging.error()`
- Return user-friendly error messages from tools
- Handle token limit errors with `is_token_limit_exceeded()` helper

```python
try:
    result = await some_async_operation()
except TokenLimitError as e:
    logging.exception("Token limit exceeded in operation")
    return f"Error: {e}"
except Exception as e:
    logging.error(f"Unexpected error: {e}")
str    raise
```

### Async/Await Patterns
- All LangGraph node functions must be `async def`
- Use `asyncio.gather()` for parallel operations
- Always use `await` with LangChain model invocations (`.ainvoke()`)
- Use `asyncio.wait_for()` with timeout for external API calls

```python
# Parallel execution
tasks = [
    research_agent.ainvoke(config)
    for agent in agents
]
results = await asyncio.gather(*tasks)

# Timeout handling
result = await asyncio.wait_for(
    model.ainvoke(messages),
    timeout=60.0
)
```

### LangGraph State Management
- Use TypedDict for state definitions
- Use `Annotated[list, reducer]` for message/history fields
- Implement `override_reducer` for fields that should replace rather than append
- State classes should inherit from `MessagesState` when messages are included

```python
from operator import add
from typing import Annotated

def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return add(current_value, new_value)

class AgentState(MessagesState):
    messages: Annotated[list[Message], override_reducer]
    research_brief: Optional[str]
```

### Pydantic Models
- Use for configuration and structured outputs
- Add `Field()` with descriptions for UI metadata
- Use `default=...` for required fields with fallbacks
- Mark optional fields with `optional=True`

```python
from pydantic import BaseModel, Field

class Configuration(BaseModel):
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations",
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "min": 1,
                "max": 100
            }
        }
    )
```

### Docstrings
- Use Google-style docstrings (ruff D401 enforces imperative mood)
- Include Args, Returns, Raises sections when applicable
- Keep docstrings concise but informative
- Add docstrings to all public functions and classes

```python
async def tavily_search(
    queries: list[str],
    max_results: int = 5,
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results per query
        config: Runtime configuration for API keys

    Returns:
        Formatted string containing summarized search results

    Raises:
        ValueError: If API key is not configured
    """
    ...
```

### Constants and Configuration
- Store API URLs, model names, and limits as module-level constants
- Use enums for fixed sets of values
- Configure models using `init_chat_model()` from LangChain
- Use `Configuration.from_runnable_config()` for runtime config

### Tool Development
- Use `@tool()` decorator from LangChain for function tools
- Add clear descriptions for tool discovery
- Use `StructuredTool` for complex tools with schema validation
- Handle errors gracefully and return error messages as strings

```python
from langchain_core.tools import tool

@tool(description="Search the web for information")
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return formatted results."""
    try:
        results = await search_api(query, max_results)
        return format_results(results)
    except Exception as e:
        return f"Search failed: {e}"
```

## Project Architecture

### Graph Structure
- **Main Graph**: `deep_researcher` in `deepresearcher.py`
  - `clarify_with_user` → `write_research_brief` → `research_supervisor` → `final_report_generation`
- **Supervisor Subgraph**: Manages research delegation
  - `supervisor` → `supervisor_tools` (循环)
- **Researcher Subgraph**: Executes individual research tasks
  - `researcher` → `researcher_tools` → `compress_research`

### Key Modules
- `deepresearcher.py`: Main graph and node functions
- `configuration.py`: Pydantic config models
- `state.py`: TypedDict state definitions
- `prompts.py`: System prompts for LLMs
- `utils.py`: Tool wrappers, API key management, MCP integration

### LLM Integration
- Use `init_chat_model()` for flexible model configuration
- Models are configured via `Configuration` class
- Support for OpenAI, Anthropic, DeepSeek, and others
- API keys from environment or runtime config

## Common Patterns

### Adding a New Tool
1. Define tool function in `utils.py` or new module
2. Use `@tool()` decorator with clear description
3. Add to `get_all_tools()` in `utils.py`
4. Update prompts if tool needs special instructions

### Adding Configuration
1. Add field to `Configuration` class in `configuration.py`
2. Include `Field()` with default and description
3. Add `x_oap_ui_config` metadata for UI integration
4. Use `from_runnable_config()` to access in nodes

### State Updates
- Use `Command(goto=..., update=...)` for transitions
- Use `{"type": "override", "value": [...]}` for replacement
- Default is append via `operator.add`

## Environment Variables

Required for development:
- `TAVILY_API_KEY`: For Tavily search
- `ARK_API_KEY`: For VolcEngine models
- `DEEPSEEK_API_KEY`: For DeepSeek models
- `GET_API_KEYS_FROM_CONFIG`: Set "true" for production deployment

See `.env.example` for reference.
