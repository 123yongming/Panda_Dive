# Panda_Dive - Agent Development Guidelines

## Build & Test Commands

```bash
# Run all tests
python -m pytest

# Run specific test
python -m pytest src/test_ark_model.py

# Run tests with verbose output
python -m pytest -v

# Run tests with coverage
python -m pytest --cov=Panda_Dive

# Linting with ruff
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking with mypy (optional dev dependency)
mypy src/Panda_Dive/
```

## Project Architecture

Panda_Dive is a LangGraph-based multi-agent deep research system with three main components:

1. **Main Graph**: Orchestrates the overall research workflow
   - `clarify_with_user`: Optional clarification phase
   - `write_research_brief`: Generates research brief
   - `research_supervisor`: Delegates to researchers (subgraph)
   - `final_report_generation`: Synthesizes final report

2. **Supervisor Subgraph**: Manages research delegation
   - `supervisor`: Plans research strategy
   - `supervisor_tools`: Executes ConductResearch tool calls

3. **Researcher Subgraph**: Executes individual research tasks
   - `researcher`: Conducts research using tools
   - `researcher_tools`: Executes search and think tools
   - `compress_research`: Synthesizes findings

## Code Style Guidelines

### Python Version & Type Hints
- Use Python 3.10+ syntax for type hints
- Prefer built-in types: `list[str]`, `dict[str, Any]` over `List[str]`, `Dict[str, Any]`
- Always use `|` for union types: `str | None` instead of `Optional[str]`
- All graph nodes must be async functions

### Import Organization

import asyncio
import logging
from datetime import datetime
from typing import Annotated, Any, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from .configuration import Configuration
from .utils import create_chat_model
```

### Naming Conventions
- **Functions**: `snake_case` - e.g., `clarify_with_user`, `create_chat_model`
- **Classes**: `PascalCase` - e.g., `Configuration`, `SearchAPI`, `ResearcherState`
- **Constants**: `UPPER_SNAKE_CASE` - e.g., `MODEL_TOKEN_LIMITS`, `TAVILY_SEARCH_DESCRIPTION`
- **Private functions**: `_leading_underscore` - e.g., `_score_single`, `_parse_search_results`
- **Graph nodes**: `snake_case` - all LangGraph node functions

### Docstrings (Google Style)
```python
async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command:
    """分析用户消息,如果研究范围不明确,则提出澄清问题。

    该函数判断用户的请求是否需要在继续研究之前进行澄清。
    如果禁用澄清或不需要澄清,则直接进入研究阶段。

    参数:
        state: 当前代理状态,包含用户消息
        config: 运行时配置,包含模型设置和偏好

    返回:
        Command:要么以澄清问题结束,要么继续撰写研究简报
    """
```

### LangGraph Patterns
- All all graph nodes return `Command` objects with `goto` and optional `update`
- Use `Command(goto="node_name")` for routing
- Use `Command(goto=END, update={"key": value})` for terminal nodes
- State classes inherit from `MessagesState` for message handling
- Use `Annotated[list, override_reducer]` for controlled state updates
- Compile graphs with `builder.compile()` and optional `config_schema`

### Configuration with Pydantic
```python
class Configuration(BaseModel):
    """DeepResearch 全局配置类。"""

    search_api: SearchAPI = Field(default=SearchAPI.TAVILY)
    max_researcher_iterations: int = Field(default=6)
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "Configuration":
        """Extract from LangGraph runtime config."""
        configuration = config.get("configurable", {}) if config else {}
        # ... extract values from env or config
```

### Structured Output Handling
```python
# Check model support for structured output
if supports_structured_output(model_name):
    model = base_model.with_structured_output(MySchema).with_retry(
        stop_after_attempt=3
    )
else:
    # Fallback to JSON parsing
    model = base_model.with_retry(stop_after_attempt=3)
    # ... parse json.loads(response.content)
```

### Error Handling
- Use async/await patterns consistently
- Log exceptions with `logging.exception()` or `logging.warning()`
- Provide user-friendly error messages in return values
- Use try/except for model calls and external API interactions
- Token limit errors: use `is_token_limit_exceeded()` helper

### Async Patterns
```python
# Parallel execution with asyncio.gather
tasks = [func(arg) for arg in args]
results = await asyncio.gather(*tasks)

# Timeout handling
try:
    result = await asyncio.wait_for(model.ainvoke(messages), timeout=60.0)
except asyncio.TimeoutError:
    logging.warning("Operation timed out")
```

### State Management
- Use `MessagesState` for message history
- Define custom state classes inheriting from `MessagesState`
- Use `override_reducer` for overriding state values instead of appending
- Message types: `HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage`

### Tool Definitions
```python
@tool(description="Tool description")
async def my_tool(param: str, config: RunnableConfig = None) -> str:
    """Tool docstring for LLM visibility."""
    # Implementation
    return result
```

### Constants & Configuration
- Environment variable checks: `os.getenv("KEY_NAME")`
- Default model parameters in `MODEL_TOKEN_LIMITS` dictionary
- Search API enum for type safety: `SearchAPI.TAVILY`

## File Structure
```
src/Panda_Dive/
├── __init__.py           # Package exports
├── deepresearcher.py     # Main graph orchestration (863 lines)
├── configuration.py       # Pydantic config models
├── state.py              # TypedDict state definitions
├── prompts.py            # System prompts
├── utils.py              # Tool wrappers & helpers
└── retrieval_quality.py   # Search quality scoring
```

## Testing Guidelines
- Test files in `src/` directory with `test_` prefix
- Use `pytest` and `asyncio.run()` for async tests
- Include basic connectivity tests for model APIs
- Mock external API calls for unit tests

## Model Support
- Models without structured output support (e.g., "ark*"): Parse JSON responses
- Always check `supports_structured_output()` before using `.with_structured_output()`
- Use `create_chat_model()` helper for model initialization
- Handle model-specific base URLs through `get_init_chat_model_params()`

## Important Notes
- Never suppress type errors with `as any` or `@ts-ignore`
- All LLM calls should include retry logic via `.with_retry()`
- Use `tags=["langsmith:nostream"]` for internal model calls
- LangSmith tracing is available via LANGSMITH_API_KEY env var
- MCP tools are loaded dynamically via `load_mcp_tools()`
