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

# Single test file
python -m pytest tests/test_parallel_eval.py

# Single test function
python -m pytest tests/test_parallel_eval.py::TestExtractSupervisorToolCalls

# Lint (ruff)
ruff check .
ruff check --fix .

# Type check (optional dev dependency)
mypy src/Panda_Dive/
```

## Project Map

| Component | File | Purpose |
|-----------|------|---------|
| **Main graph** | `src/Panda_Dive/deepresearcher.py` | Graph orchestration, supervisor & researcher subgraphs |
| **Configuration** | `src/Panda_Dive/configuration.py` | `Configuration` model, `SearchAPI` enum, validation |
| **State** | `src/Panda_Dive/state.py` | TypedDict states with `Annotated[...]` reducers |
| **Prompts** | `src/Panda_Dive/prompts.py` | System prompts and templates |
| **Utils** | `src/Panda_Dive/utils.py` | `create_chat_model()`, search APIs, MCP loading |
| **Retrieval** | `src/Panda_Dive/retrieval_quality.py` | Query rewriting, scoring, reranking |
| **Evaluators** | `tests/evaluators.py` | LLM-based evaluators for research quality |
| **Tests** | `tests/test_*.py` | Unit tests for evaluators and components |

## Code Style

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
- Google-style docstrings (English required)
- Chinese comments acceptable inline

## LangGraph Patterns

- Nodes return `Command(goto=..., update={...})`
- Use `Command(goto=END, update={...})` for terminal nodes
- State classes inherit `MessagesState`
- List fields use `Annotated[list[T], override_reducer]`
- Use `create_chat_model()` from utils.py (never `init_chat_model` directly)
- LLM calls chain `.with_retry(stop_after_attempt=3)`
- Internal calls use `tags=["langsmith:nostream"]`

## Error Handling

- Use `try/except` around model calls and external APIs
- Log with `logging.warning()` or `logging.exception()`
- Use `is_token_limit_exceeded()` for token-limit handling

## Async Patterns

- Use `asyncio.gather` for parallel tasks
- Use `asyncio.wait_for` for timeouts
- Use `asyncio.to_thread` for blocking work

## Configuration

- Pydantic `BaseModel` with defaults and validation
- Load runtime config via `Configuration.from_runnable_config(config)`
- Environment variable checks via `os.getenv(...)`

## Testing

- Tests live under `tests/` with `test_` prefix
- Prefer `pytest` and `asyncio.run()` for async tests
- Mock external API calls for unit tests

## Anti-Patterns

- Never use `Optional[T]` - always `T | None`
- Never call `deep_researcher.invoke()`; use async `ainvoke()`
- Never mutate state directly; always return `Command(update={...})`
- Never import or use `init_chat_model` directly
- Never add sync graph nodes
- Do not concatenate state lists without `override_reducer`

## Tooling (pyproject.toml)

- Ruff lint: `E`, `F`, `I`, `D`, `D401`, `T201`, `UP`
- Ruff ignore: `UP006`, `UP007`, `UP035`, `D417`, `E501`
- Pydocstyle: Google convention
- pytest: `--ignore=nul`
- Build: `setuptools` + `wheel`

## Evaluation

```bash
# Smoke test (2 examples)
python tests/run_evaluate.py --smoke --dataset-name "deep_research_bench"
python tests/run_evaluate.py --smoke --model openai:gpt-4o

# Supervisor parallelism evaluation
python tests/run_evaluate.py \
  --dataset-name "Panda_Dive: Supervisor Parallelism" \
  --max-concurrency 1 \
  --experiment-prefix "supervisor-parallel"

# Full evaluation (expensive)
python tests/run_evaluate.py --full

# Export results
python tests/extract_langsmith_data.py \
  --project-name "deep-research-eval-..." \
  --model-name "gpt-4o" \
  --output-dir tests/expt_results/
```

## Cursor / Copilot Rules

No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.
