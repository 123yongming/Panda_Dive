# Evaluation Guide

## Goal
Provide a repeatable Phase 1 evaluation flow with quality, cost, and latency reporting.

## Prerequisites
1. Set `LANGSMITH_API_KEY`.
2. Install dependencies:
   - `pip install -e .`
   - `pip install -e ".[dev]"`

## 1) Run smoke evaluation
```bash
python tests/run_evaluate.py --smoke --dataset-name "deep_research_bench"
```

## 2) Export run outputs
```bash
python tests/extract_langsmith_data.py \
  --project-name "<langsmith-project>" \
  --model-name "<model-name>" \
  --output-dir tests/expt_results/
```

## 3) Generate benchmark card
```bash
python tests/generate_benchmark_card.py \
  --project-name "<langsmith-project>" \
  --dataset-name "deep_research_bench" \
  --model-name "<model-name>" \
  --output docs/benchmark-card-v1.md
```

## 4) Core local regression gate
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

