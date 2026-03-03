"""Tests for retrieval quality module."""

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
    rewritten = await rewrite_query_for_retrieval("test query", {}, model, config)
    assert len(rewritten) == 2


@pytest.mark.anyio
async def test_rerank_results_respects_top_k():
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


def test_parse_search_results_empty_input_returns_empty_list():
    assert _parse_search_results("") == []

