# Panda_Dive - Agent Development Guidelines

These notes are for agentic coding tools operating in this repo. Keep changes small, follow existing patterns, and prefer async-first LangGraph conventions.

## Build, Lint, Test

```bash
# Install (editable)
pip install -e .
pip install -e ".[dev]"

# Alternative install with uv
uv sync
uv pip install -r pyproject.toml

# Tests
python -m pytest
python -m pytest -v
python -m pytest --cov=Panda_Dive

# Single test
python -m pytest src/test_ark_model.py
python -m pytest src/test_api.py::test_function_name

# Lint (ruff)
ruff check .
ruff check --fix .

# Type check (optional dev dependency)
mypy src/Panda_Dive/
```

## Project Map

- `src/Panda_Dive/deepresearcher.py` main graph orchestration, subgraphs compiled near file bottom
- `src/Panda_Dive/configuration.py` configuration model, search API enum, validation
- `src/Panda_Dive/state.py` TypedDict state definitions with reducers
- `src/Panda_Dive/prompts.py` system prompts and templates
- `src/Panda_Dive/utils.py` tool wrappers, model helpers, MCP loading
- `src/Panda_Dive/retrieval_quality.py` query rewriting, scoring, reranking
- `src/Panda_Dive/__init__.py` package exports

## Code Style and Conventions

### Python and Types
- Python 3.10+ syntax only
- Prefer built-ins: `list[str]`, `dict[str, Any]`
- Use `T | None`, never `Optional[T]`
- All LangGraph nodes are `async def` and return `Command`

### Import Order
Standard library, third-party, local modules. Keep grouped and sorted.

```python
import asyncio
import logging
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from .configuration import Configuration
from .utils import create_chat_model
```

### Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: `_leading_underscore`
- Graph nodes: `snake_case`

### Docstrings and Comments
- Google-style docstrings
- English docstrings required; Chinese comments acceptable

### LangGraph Patterns
- Nodes return `Command(goto=..., update={...})`
- Use `Command(goto=END, update={...})` for terminal nodes
- State classes inherit `MessagesState`
- List fields use `Annotated[list[T], override_reducer]`

### Models and Structured Output
- Use `create_chat_model()` from `utils.py` (do not call `init_chat_model` directly)
- Check `supports_structured_output()` before `.with_structured_output(...)`
- LLM calls chain `.with_retry(stop_after_attempt=3)`
- Internal calls use `tags=["langsmith:nostream"]`

### Error Handling
- Use `try/except` around model calls and external API interactions
- Log with `logging.warning()` or `logging.exception()`
- Use `is_token_limit_exceeded()` for token-limit handling

### Async Patterns
- Use `asyncio.gather` for parallel tasks
- Use `asyncio.wait_for` for timeouts
- Use `asyncio.to_thread` for blocking work

### Configuration
- Pydantic `BaseModel` with defaults and validation
- Load runtime config via `Configuration.from_runnable_config(config)`
- Environment variable checks via `os.getenv(...)`

## Testing Guidelines

- Tests live under `src/` with `test_` prefix
- Prefer `pytest` and `asyncio.run()` for async tests
- Mock external API calls for unit tests

## Anti-Patterns

- Never use `Optional[T]`
- Never call `deep_researcher.invoke()`; use async `ainvoke()`
- Never mutate state directly; always return `Command(update={...})`
- Never import or use `init_chat_model` directly
- Never add sync graph nodes
- Do not concatenate state lists without `override_reducer`

## Tooling Configuration (pyproject.toml)

- Ruff lint selects: `E`, `F`, `I`, `D`, `D401`, `T201`, `UP`
- Ruff ignores: `UP006`, `UP007`, `UP035`, `D417`, `E501`
- Pydocstyle convention: Google
- pytest addopts: `--ignore=nul`
- Build system: `setuptools` + `wheel`

## Evaluation (Optional, Costly)

```bash
# Smoke test (2 examples)
python tests/run_evaluate.py --smoke --dataset-name "deep_research_bench"
python tests/run_evaluate.py --smoke --model openai:gpt-4o
python tests/run_evaluate.py --smoke --max-concurrency 2 --timeout-seconds 1800

# Full evaluation (expensive)
python tests/run_evaluate.py --full
python tests/run_evaluate.py --full --model anthropic:claude-3-5-sonnet-20241022
python tests/run_evaluate.py --full --dataset-name "Custom Dataset" --experiment-prefix "my-experiment"

# Export results
python tests/extract_langsmith_data.py \
  --project-name "deep-research-eval-smoke-20250204-120000" \
  --model-name "gpt-4o" \
  --output-dir tests/expt_results/
```

## Cursor / Copilot Rules

No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found in this repo.
