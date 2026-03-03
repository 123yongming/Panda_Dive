# Phase 1 Acceptance Checklist

## Targets from `future-direction.md`
1. Core unit/regression tests for retrieval chain, reducer, and routing.
2. First public benchmark card format (quality/cost/latency).
3. Unified model initialization/configuration entry.
4. Documentation entry points for architecture, evaluation, and contribution.
5. CI gate for repeatable regression checks.

## Completion Status
- [x] Core regression tests added under `tests/`.
- [x] Retrieval quality tests standardized in `tests/test_retrieval_quality.py`.
- [x] `init_chat_model` direct usage removed from `deepresearcher.py`.
- [x] Benchmark card generator added: `tests/generate_benchmark_card.py`.
- [x] Benchmark card V1 file added: `docs/benchmark-card-v1.md`.
- [x] CI workflow added: `.github/workflows/ci.yml`.
- [x] Docs index and entry docs added under `docs/`.

## Exit Condition Evidence
- Core tests are executable with a fixed command set.
- Lint gate is executable via `ruff check src/Panda_Dive`.
- Benchmark card generation command is documented and scripted.

