# Panda_Dive Architecture Overview

## Scope
This document is the Phase 1 architecture entry point for contributors.

## Graph Layers
1. Main graph: user clarification, research brief generation, final report synthesis.
2. Supervisor subgraph: task delegation, concurrency control, aggregation.
3. Researcher subgraph: tool calling loop, retrieval-quality enhancement, compression.

## Key Modules
- `src/Panda_Dive/deepresearcher.py`: graph construction and node orchestration.
- `src/Panda_Dive/configuration.py`: runtime config model and defaults.
- `src/Panda_Dive/state.py`: state schemas and reducer behavior.
- `src/Panda_Dive/retrieval_quality.py`: query rewrite, relevance score, rerank.
- `src/Panda_Dive/utils.py`: model factory, search tools, MCP integration.

## Phase 1 Constraints
1. Async-only execution path for graph invocation (`ainvoke`).
2. Single model initialization entry (`create_chat_model` in `utils.py`).
3. Core regression checks for retrieval, reducer, and routing behavior.
4. CI gate on lint + core regression tests.

## Related Docs
- See [architecture-diagrams.md](architecture-diagrams.md) for flow diagrams.
- See [state-passing.md](state-passing.md) for state transition examples.

