# Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Phase 1 on branch `trust-building` with test gates, CI gates, benchmark card workflow, and docs entry points.

**Architecture:** Build a stable quality gate around existing LangGraph architecture without introducing Phase 2 features. Keep scope focused on repeatability (tests/CI), observability of outcomes (benchmark card), and contributor onboarding docs.

**Tech Stack:** Python, pytest, ruff, GitHub Actions, LangSmith scripts.

---

### Task 1: Core Regression Tests

**Files:**
- Create: `tests/test_retrieval_quality.py`
- Create: `tests/test_state_reducer.py`
- Create: `tests/test_tool_routing.py`
- Create: `tests/test_model_initialization_entry.py`
- Remove: `src/test_retrieval_quality.py`

**Step 1:** Add tests for retrieval quality chain behavior.
**Step 2:** Add reducer and tool routing tests.
**Step 3:** Add convention test to block direct `init_chat_model` in graph.
**Step 4:** Run targeted tests and ensure pass.

### Task 2: Model Entry Unification

**Files:**
- Modify: `src/Panda_Dive/deepresearcher.py`

**Step 1:** Remove direct `init_chat_model` import and dead `configurable_model`.
**Step 2:** Re-run model-entry convention test.

### Task 3: Benchmark Card Workflow

**Files:**
- Create: `src/Panda_Dive/benchmark_card.py`
- Create: `tests/test_benchmark_card.py`
- Create: `tests/generate_benchmark_card.py`
- Create: `docs/benchmark-card-v1.md`

**Step 1:** Add failing tests for metric aggregation and markdown generation.
**Step 2:** Implement benchmark summary module.
**Step 3:** Add card generation CLI script.
**Step 4:** Verify tests pass.

### Task 4: CI Gate

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1:** Add workflow for `push` and `pull_request`.
**Step 2:** Enforce `ruff check src/Panda_Dive`.
**Step 3:** Enforce core regression test command.

### Task 5: Documentation Entrypoints

**Files:**
- Create: `docs/README.md`
- Create: `docs/architecture.md`
- Create: `docs/evaluation-guide.md`
- Create: `docs/contributing.md`
- Create: `docs/phase1-acceptance.md`
- Modify: `README.md`

**Step 1:** Add docs index and cross-links.
**Step 2:** Add architecture/evaluation/contributing entry docs.
**Step 3:** Fix README quick-start to async `ainvoke`.
**Step 4:** Add Phase 1 acceptance checklist.

