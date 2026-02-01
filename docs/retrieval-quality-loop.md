# Retrieval Quality Loop - Technical Report

## Summary
This report documents the Phase 1 retrieval quality loop added to Panda_Dive. The loop improves search results by rewriting queries, scoring relevance, and reranking results before they enter the research compression pipeline. The implementation is integrated into the researcher tool execution path and records quality telemetry in state.

## Goals
- Improve retrieval precision and recall without changing the core research graph.
- Provide tunable parameters for query expansion and reranking.
- Preserve observability by recording quality metadata in state.
- Maintain graceful fallback behavior if structured output is unavailable.

## Architecture Overview
The loop is implemented as a standalone module plus a single integration point in the researcher tool execution path.

Core modules:
- `src/Panda_Dive/retrieval_quality.py`
- `src/Panda_Dive/deepresearcher.py` (integration)
- `src/Panda_Dive/configuration.py` (tunables)
- `src/Panda_Dive/state.py` (quality telemetry)

## Flow Diagram
```
User Query
  -> Query Rewrite (LLM)
  -> Search Tool Execution
  -> Parse Results
  -> Relevance Scoring (LLM)
  -> Rerank (score + source weight)
  -> Format Results
  -> Researcher State Update
```

## Core Components

### 1) Query Rewrite
**Implementation:** `rewrite_query_for_retrieval()` in `src/Panda_Dive/retrieval_quality.py`

Purpose:
- Generate multiple semantically aligned variants of the original query.
- Expand coverage while preserving the original query as the first variant.

Key behaviors:
- Uses structured output when supported, otherwise falls back to text parsing.
- Deduplicates variants and ensures original query stays first.

Inputs:
- `original_query: str`
- `context: dict[str, Any] | None`
- `model: BaseChatModel`

Outputs:
- `list[str]` of query variants

Config:
- `query_variants` (default 3)

### 2) Relevance Scoring
**Implementation:** `score_retrieval_quality()` in `src/Panda_Dive/retrieval_quality.py`

Purpose:
- Assign a relevance score in range `[0.0, 1.0]` for each retrieved result.

Key behaviors:
- Parallel scoring via `asyncio.gather()`.
- Structured output with fallback to numeric parsing.
- Clamps scores to `[0.0, 1.0]`.

Inputs:
- `results: list[dict[str, Any]]` with `title`, `url`, `summary`
- `query: str`

Outputs:
- Results list augmented with `score: float`

### 3) Reranking
**Implementation:** `rerank_results()` in `src/Panda_Dive/retrieval_quality.py`

Purpose:
- Sort results by weighted relevance score and return top-k.

Key behaviors:
- Optionally calls scoring when scores are missing.
- Applies source credibility weighting (default: boosts `.gov` / `.edu`).
- Returns top-k results.

Config:
- `rerank_top_k` (default 10)
- `rerank_weight_source` (auto|low|medium|high)

### 4) Integration Point
**Implementation:** `researcher_tools()` in `src/Panda_Dive/deepresearcher.py`

Integration steps:
1. Rewrite queries before tool execution.
2. Execute search tools with expanded queries.
3. Parse tool output into structured results.
4. Score and rerank results.
5. Format results back into tool output format.
6. Record quality telemetry in state.

## Configuration (Phase 1 Tunables)
Defined in `src/Panda_Dive/configuration.py`:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `query_variants` | int | 3 | Number of query variants to generate |
| `relevance_threshold` | float | 0.7 | Minimum relevance score (not enforced in code yet) |
| `rerank_top_k` | int | 10 | Number of results returned after rerank |
| `rerank_weight_source` | str | auto | Source weighting strategy |

## State Telemetry
Defined in `src/Panda_Dive/state.py` (ResearcherState):

| Field | Type | Purpose |
| --- | --- | --- |
| `rewritten_queries` | `list[str]` | All query variants used |
| `relevance_scores` | `list[dict[str, float]]` | URL + relevance score |
| `reranked_results` | `list[str]` | Final URL ordering |
| `quality_notes` | `list[str]` | Human-readable quality log |

## Error Handling and Fallbacks
- Structured output is used when model supports it; otherwise, text parsing is used.
- Scoring and rewrite steps catch exceptions and default to safe fallbacks.
- Tool execution is wrapped to avoid halting the research loop on tool errors.

## Tests
**File:** `src/test_retrieval_quality.py`

Coverage:
- Scoring returns bounded scores.
- Query rewriting respects `query_variants`.
- Reranking respects `rerank_top_k`.
- Integration test verifies that quality telemetry is populated.

## Extensibility Notes
- Add new quality signals by extending `retrieval_quality.py` and appending to `quality_notes`.
- Use `Configuration.from_runnable_config(config)` to access new tunables.
- Update `ResearcherState` to store new telemetry fields when needed.

## Known Limitations
- `relevance_threshold` is defined but not currently enforced as a hard filter.
- Source weighting is URL-based and does not evaluate domain reputation beyond `.gov` / `.edu` heuristics.

## References
- Query transformations and multi-query retrieval: https://www.blog.langchain.com/query-transformations/
- Contextual compression and reranking: https://python.langchain.com/docs/how_to/contextual_compression
- Advanced RAG techniques (reranking and CRAG): https://neo4j.com/blog/genai/advanced-rag-techniques/
