# Contributing

## Setup
```bash
pip install -e .
pip install -e ".[dev]"
```

## Branch and PR expectations
1. Keep changes scoped to one objective.
2. Add tests for behavior changes.
3. Update docs when workflow or architecture changes.

## Required checks (Phase 1 gate)
```bash
ruff check src/Panda_Dive
python -m pytest -q \
  tests/test_parallel_eval.py \
  tests/test_retrieval_quality.py \
  tests/test_state_reducer.py \
  tests/test_tool_routing.py \
  tests/test_model_initialization_entry.py \
  tests/test_benchmark_card.py
```

## CI
- Workflow file: `.github/workflows/ci.yml`
- CI runs lint + core regression tests on push and pull request.

