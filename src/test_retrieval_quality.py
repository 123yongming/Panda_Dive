import types

import pytest

from Panda_Dive.deepresearcher import researcher_tools
from Panda_Dive.retrieval_quality import (
    rerank_results,
    rewrite_query_for_retrieval,
    score_retrieval_quality,
)


class FakeModel:
    def __init__(self, score: float = 0.8, queries: list[str] | None = None):
        self._score = score
        self._queries = queries or ["query one", "query two", "query three"]
        self._schema = None

    def with_structured_output(self, schema):
        self._schema = schema
        return self

    def with_retry(self, stop_after_attempt: int = 1):
        return self

    async def ainvoke(self, messages):
        if self._schema and self._schema.__name__ == "_RewriteOutput":
            return types.SimpleNamespace(queries=self._queries)
        return types.SimpleNamespace(score=self._score)


class FakeTool:
    name = "tavily_search"

    async def ainvoke(self, args, config):
        return (
            "Search results: \n\n"
            "\n\n--- SOURCE 1: Example One ---\n"
            "URL: https://example.com/one\n\n"
            "SUMMARY:\nSummary one\n\n"
            "\n\n" + "-" * 80 + "\n"
        )


class FakeMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


@pytest.mark.anyio
async def test_score_retrieval_quality_returns_scores():
    model = FakeModel(score=0.6)
    results = [{"title": "Doc", "url": "https://example.com", "summary": "Text"}]
    scored = await score_retrieval_quality(results, "test query", model, {})
    assert len(scored) == 1
    assert scored[0]["score"] == 0.6


@pytest.mark.anyio
async def test_rewrite_query_for_retrieval_respects_variants():
    model = FakeModel(queries=["alpha", "beta", "gamma"])
    config = {"configurable": {"query_variants": 2, "max_structured_output_retries": 1}}
    rewritten = await rewrite_query_for_retrieval("base", None, model, config)
    assert len(rewritten) == 2
    assert rewritten[0] == "base"


@pytest.mark.anyio
async def test_rerank_results_respects_top_k():
    model = FakeModel()
    results = [
        {"title": "A", "url": "https://example.com/a", "summary": "x", "score": 0.2},
        {"title": "B", "url": "https://example.com/b", "summary": "y", "score": 0.9},
    ]
    config = {
        "configurable": {
            "rerank_top_k": 1,
            "rerank_weight_source": "low",
            "max_structured_output_retries": 1,
        }
    }
    reranked = await rerank_results(results, "test", model, config)
    assert len(reranked) == 1
    assert reranked[0]["url"].endswith("/b")


@pytest.mark.anyio
async def test_researcher_tools_integration(monkeypatch):
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")

    async def fake_get_all_tools(_config):
        return [FakeTool()]

    def fake_create_chat_model(*args, **kwargs):
        return FakeModel(score=0.75, queries=["alt one", "alt two"])

    monkeypatch.setattr("Panda_Dive.deepresearcher.get_all_tools", fake_get_all_tools)
    monkeypatch.setattr(
        "Panda_Dive.deepresearcher.create_chat_model", fake_create_chat_model
    )

    tool_calls = [
        {"name": "tavily_search", "args": {"queries": ["base query"]}, "id": "1"}
    ]
    state = {
        "researcher_messages": [FakeMessage(tool_calls)],
        "tool_call_iterations": 0,
        "research_topic": "topic",
    }
    config = {
        "configurable": {
            "query_variants": 2,
            "rerank_top_k": 2,
            "rerank_weight_source": "low",
            "max_structured_output_retries": 1,
            "research_model": "deepseek-chat",
            "research_model_max_tokens": 1024,
        }
    }
    result = await researcher_tools(state, config)
    assert result.update["rewritten_queries"]
    assert result.update["relevance_scores"]
    assert result.update["reranked_results"]
    assert result.update["quality_notes"]
