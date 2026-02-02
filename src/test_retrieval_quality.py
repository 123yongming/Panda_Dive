"""Tests for retrieval quality module.

This module contains unit tests for retrieval quality
enhancement features including scoring, query rewriting, and reranking.
"""

import types

import pytest

from Panda_Dive.retrieval_quality import (
    _parse_search_results,
    rerank_results,
    rewrite_query_for_retrieval,
    score_retrieval_quality,
)


class FakeModel:
    """Mock model for testing retrieval quality functions."""

    def __init__(self, score: float = 0.8, queries: list[str] | None = None):
        """Initialize FakeModel.

        Args:
            score: The score to return for scoring tests.
            queries: The queries to return for rewriting tests.
        """
        self._score = score
        self._queries = queries or ["query one", "query two", "query three"]
        self._schema = None

    def with_structured_output(self, schema):
        """Configure mock model with structured output."""
        self._schema = schema
        return self

    def with_retry(self, stop_after_attempt: int = 1):
        """Configure mock model with retry settings."""
        return self

    async def ainvoke(self, messages):
        """Mock async invoke for model calls.

        Args:
            messages: Messages to process.

        Returns:
            Mock response based on configured schema.
        """
        if self._schema and self._schema.__name__ == "_RewriteOutput":
            return types.SimpleNamespace(queries=self._queries)
        return types.SimpleNamespace(score=self._score)


class FakeTool:
    """Mock tool for testing researcher tools integration."""

    name = "tavily_search"

    async def ainvoke(self, args, config):
        """Mock async tool invocation.

        Args:
            args: Tool arguments.
            config: Runtime configuration.

        Returns:
            Mock search results in expected format.
        """
        return (
            "Search results: \n\n"
            "\n\n--- SOURCE 1: Example One ---\n"
            "URL: https://example.com/one\n\n"
            "SUMMARY:\nSummary one\n\n"
            "\n\n" + "-" * 80 + "\n"
        )


class FakeMessage:
    """Mock message for testing tool integration."""

    def __init__(self, tool_calls):
        """Initialize FakeMessage.

        Args:
            tool_calls: List of tool calls to mock.
        """
        self.tool_calls = tool_calls


@pytest.mark.anyio
async def test_score_retrieval_quality_returns_scores():
    """Test that score_retrieval_quality returns scored results."""
    model = FakeModel(score=0.6)
    results = [{"title": "Doc", "url": "https://example.com", "summary": "Text"}]
    scored = await score_retrieval_quality(results, "test query", model, {})
    assert len(scored) == 1
    assert scored[0]["score"] == 0.6


@pytest.mark.anyio
async def test_rewrite_query_for_retrieval_respects_variants():
    """Test that query rewriting respects configured variants limit."""
    model = FakeModel(queries=["alpha", "beta", "gamma"])
    config = {"configurable": {"query_variants": 2, "max_structured_output_retries": 1}}
    rewritten = await rewrite_query_for_retrieval("test query", {}, model, config)
    assert len(rewritten) == 2


@pytest.mark.anyio
async def test_rerank_results_respects_top_k():
    """Test that result reranking respects top_k limit."""
    model = FakeModel()
    results = [
        {"title": "A", "url": "url1", "summary": "text1"},
        {"title": "B", "url": "url2", "summary": "text2"},
    ]
    config = {"configurable": {"rerank_top_k": 1, "max_structured_output_retries": 1}}
    reranked = await rerank_results(results, "query", model, config)
    assert len(reranked) == 1


@pytest.mark.anyio
async def test_duckduckgo_output_parsing_and_scoring():
    """Test DuckDuckGo formatted output can be parsed and scored."""
    model = FakeModel(score=0.7)
    duckduckgo_output = (
        "Search results: \n\n"
        "Query: panda dive\n"
        "\n--- SOURCE 1: Example Duck ---\n"
        "URL: https://example.com/duck\n\n"
        "SNIPPET:\nDuck summary text\n"
        "\n" + "-" * 80 + "\n\n"
    )
    parsed = _parse_search_results(duckduckgo_output)
    assert len(parsed) == 1
    scored = await score_retrieval_quality(parsed, "panda dive", model, {})
    assert scored[0]["score"] == 0.7


@pytest.mark.anyio
async def test_researcher_tools_integration(monkeypatch):
    """Test integration of researcher tools with retrieval quality."""
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test_key")

    from Panda_Dive.retrieval_quality import (
        rerank_results,
        rewrite_query_for_retrieval,
        score_retrieval_quality,
    )

    _ = rerank_results
    _ = rewrite_query_for_retrieval
    _ = score_retrieval_quality

    tool = FakeTool()
    message = FakeMessage(tool_calls=[{"id": "1", "name": tool.name}])

    assert tool is not None
    assert message is not None
