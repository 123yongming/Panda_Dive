# Evaluation (Deep Research Bench)

This directory contains the evaluation scripts migrated from Open Deep Research, adapted to Panda_Dive and Ark-based models.

## Prerequisites
- Python dependencies already in `pyproject.toml`: `langsmith`, `python-dotenv`, `langgraph`, `langchain-core`.
- Environment variables in `.env` (never commit keys):
  - `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (defaults to `open_deep_research_local`)
  - `TAVILY_API_KEY`
  - `ARK_API_KEY` (or `DEEPSEEK_API_KEY` if you use DeepSeek)
  - `RESEARCH_MODEL`, `SUMMARIZATION_MODEL`, `COMPRESSION_MODEL`, `FINAL_REPORT_MODEL`
  - `*_MAX_TOKENS`
  - `EVAL_MODEL` (optional), `EVAL_MODEL_MAX_TOKENS` (optional)

## Run Evaluation
```bash
python tests/run_evaluate.py
```

High-fidelity small sample:
```bash
python tests/run_evaluate.py --limit 5
```

Low-cost smoke:
```bash
python tests/run_evaluate.py --limit 5 --smoke
```

## Export JSONL Results
```bash
python tests/extract_langsmith_data.py \
  --project-name "<LANGSMITH_PROJECT>" \
  --dataset-name "Deep Research Bench" \
  --model-name "${FINAL_REPORT_MODEL}"
```

Outputs JSONL to `tests/expt_results/{dataset_name}_{model_name}.jsonl`:
```jsonl
{"id":"...","prompt":"...","article":"..."}
```

## Notes
- `run_evaluate.py` resolves the Deep Research Bench dataset from the public URL and clones it if needed (dataset name: `deep_research_bench`).
- Evaluators reuse Panda_Dive model initialization; no OpenAI/Anthropic hardcoding is used.
- `pairwise_evaluation.py` is intentionally left as a non-migrated stub.
- `supervisor_parallel_evaluation.py` is a minimal runnable version; adjust state access if your graph state differs.

## Troubleshooting
- LangSmith auth error: verify `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`.
- Dataset not found: the script auto-clones the public dataset; ensure network access to LangSmith.
- Model call failures: verify model env vars and keys (`ARK_API_KEY`, `DEEPSEEK_API_KEY`).
