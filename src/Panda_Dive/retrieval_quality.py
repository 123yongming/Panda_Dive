"""Retrieval quality enhancement module for Panda_Dive.

This module provides utilities for improving search result quality through
scoring, query rewriting, and result reranking. Part of Phase 1 retrieval
quality loop.
"""

import asyncio
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from .configuration import Configuration
from .utils import supports_structured_output


class _ScoreOutput(BaseModel):
    """Structured output for relevance scoring."""

    score: float = Field(ge=0.0, le=1.0)


class _RewriteOutput(BaseModel):
    """Structured output for query rewriting."""

    queries: list[str]


def _clamp_score(score: float) -> float:
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _extract_first_float(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


async def _ainvoke_with_backoff(
    model: BaseChatModel, prompt: str, max_retries: int
) -> Any:
    retries = max(1, max_retries)
    for attempt in range(retries):
        try:
            return await model.ainvoke([HumanMessage(content=prompt)])
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == retries - 1:
                raise
            await asyncio.sleep(2**attempt)


def _normalize_query_variants(queries: list[str], original_query: str) -> list[str]:
    normalized: list[str] = []
    for item in queries:
        cleaned = item.strip()
        if not cleaned:
            continue
        if cleaned not in normalized:
            normalized.append(cleaned)
    if original_query not in normalized:
        normalized.insert(0, original_query)
    return normalized


def _parse_search_results(raw_text: str) -> list[dict[str, Any]]:
    if not raw_text:
        return []
    pattern = re.compile(
        r"--- SOURCE\s+\d+:\s+(?P<title>.*?)\s+---\s*"
        r"(?:URL:\s*(?P<url>\S*)\s*)?"
        r"(?:SUMMARY|SNIPPET):\s*(?P<summary>.*?)(?:\n\s*-{5,}|\Z)",
        re.DOTALL,
    )
    results: list[dict[str, Any]] = []
    for match in pattern.finditer(raw_text):
        results.append(
            {
                "title": match.group("title").strip(),
                "url": (match.group("url") or "").strip(),
                "summary": match.group("summary").strip(),
            }
        )
    return results


def _format_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No valid search results found. Please try different search queries or use a different search API."
    formatted_output = "Search results: \n\n"
    for i, result in enumerate(results):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        summary = result.get("summary", "")
        formatted_output += f"\n\n--- SOURCE {i + 1}: {title} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{summary}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    return formatted_output


def _source_weight(url: str, strategy: str) -> float:
    if not url:
        return 1.0
    if strategy == "low":
        return 1.0
    if strategy == "medium":
        return 1.05
    if strategy == "high":
        return 1.1
    lowered = url.lower()
    if ".gov" in lowered or ".edu" in lowered:
        return 1.1
    return 1.0


async def score_retrieval_quality(
    results: list[dict[str, Any]],
    query: str,
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> list[dict[str, Any]]:
    """Score the relevance of retrieved search results.

    Args:
        results: List of search result dictionaries containing URLs and content.
        query: Original search query string.
        model: Language model instance for quality scoring.
        config: Optional runtime configuration for scoring behavior.

    Returns:
        List of search results with an added "score" field per item.
    """
    if not results:
        return []
    configurable = Configuration.from_runnable_config(config)
    max_retries = configurable.max_structured_output_retries
    structured_supported = supports_structured_output(configurable.research_model)

    max_concurrency = max(1, configurable.max_concurrent_research_units)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_single(result: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _score_single_inner(result)

    async def _score_single_inner(result: dict[str, Any]) -> dict[str, Any]:
        title = result.get("title", "")
        summary = result.get("summary", "")
        prompt = (
            "Score the relevance of the search result to the query on a 0.0 to 1.0 scale. "
            "Return only the numeric score.\n\n"
            f"Query: {query}\n"
            f"Title: {title}\n"
            f"Summary: {summary}\n"
        )
        if structured_supported:
            try:
                scorer = model.with_structured_output(_ScoreOutput).with_retry(
                    stop_after_attempt=max_retries
                )
                response = await scorer.ainvoke([HumanMessage(content=prompt)])
                if response is None:
                    logging.warning("Structured output returned None, using fallback")
                    raise ValueError("Empty structured output response")
                score = _clamp_score(response.score)
                return {**result, "score": score}
            except Exception as structured_exc:
                logging.warning("Structured output scoring failed: %s", structured_exc)
        try:
            response = await _ainvoke_with_backoff(model, prompt, max_retries)
            if response is None:
                logging.warning("Model returned None, using default score")
                score = 0.5
            else:
                # Try to extract score from response
                score_value = None
                # First check if response has content attribute (standard message)
                if hasattr(response, "content") and response.content is not None:
                    content_str = str(response.content).strip()
                    if content_str:
                        try:
                            score_value = float(content_str)
                        except ValueError:
                            score_value = _extract_first_float(content_str)
                # If not found, check if response has score attribute (SimpleNamespace)
                if score_value is None and hasattr(response, "score"):
                    try:
                        score_value = float(response.score)
                    except (ValueError, TypeError):
                        pass
                # Last resort: try to parse string representation
                if score_value is None:
                    str_repr = str(response).strip()
                    try:
                        score_value = float(str_repr)
                    except ValueError:
                        score_value = _extract_first_float(str_repr)
                        if score_value is None:
                            logging.warning(
                                "Could not parse score from response: %s", str_repr
                            )
                            score_value = 0.5
                score = _clamp_score(score_value)
        except Exception as exc:
            logging.warning("Fallback scoring failed: %s", exc)
            score = 0.0
        return {**result, "score": score}

    tasks = [_score_single(result) for result in results]
    return await asyncio.gather(*tasks)


async def rewrite_query_for_retrieval(
    original_query: str,
    context: dict[str, Any] | None,
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> list[str]:
    """Rewrite a search query to improve retrieval effectiveness.

    Args:
        original_query: Original user query string.
        context: Optional context dictionary with previous search info.
        model: Language model instance for query optimization.
        config: Optional runtime configuration for rewriting behavior.

    Returns:
        List of optimized query strings for better search results.
    """
    configurable = Configuration.from_runnable_config(config)
    variants = configurable.query_variants
    structured_supported = supports_structured_output(configurable.research_model)
    prompt = (
        "Generate concise alternative search queries that preserve the user intent. "
        f"Return {variants} distinct queries.\n\n"
        f"Original query: {original_query}\n"
    )
    if context:
        prompt += f"Context: {context}\n"
    if structured_supported:
        try:
            rewrite_model = model.with_structured_output(_RewriteOutput).with_retry(
                stop_after_attempt=configurable.max_structured_output_retries
            )
            response = await rewrite_model.ainvoke([HumanMessage(content=prompt)])
            if response is None or not hasattr(response, "queries"):
                logging.warning("Structured output response invalid, using fallback")
                raise ValueError("Invalid structured output response")
            queries = _normalize_query_variants(response.queries, original_query)
            return queries[: max(variants, 1)]
        except Exception as structured_exc:
            logging.warning(
                "Structured output query rewriting failed: %s", structured_exc
            )
    try:
        response = await _ainvoke_with_backoff(
            model, prompt, configurable.max_structured_output_retries
        )
        if response is None:
            logging.warning("Model returned None, using original query")
            return [original_query]
        text = getattr(response, "content", str(response))
        if not text or not str(text).strip():
            logging.warning("Empty response content, using original query")
            return [original_query]
        raw_queries = [line.strip("- ") for line in str(text).splitlines()]
        queries = _normalize_query_variants(raw_queries, original_query)
        return queries[: max(variants, 1)]
    except Exception as exc:
        logging.warning("Fallback query rewriting failed: %s", exc)
        return [original_query]


async def rerank_results(
    results: list[dict[str, Any]],
    query: str,
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> list[dict[str, Any]]:
    """Rerank search results by relevance to the query.

    Args:
        results: List of search result dictionaries to rerank.
        query: Search query string for relevance comparison.
        model: Language model instance for reranking.
        config: Optional runtime configuration for reranking behavior.

    Returns:
        Reordered list of results sorted by relevance (most relevant first).
    """
    if not results:
        return []
    configurable = Configuration.from_runnable_config(config)
    scored_results = results
    if any("score" not in item for item in results):
        scored_results = await score_retrieval_quality(results, query, model, config)
    strategy = configurable.rerank_weight_source
    weighted: list[dict[str, Any]] = []
    for item in scored_results:
        url = item.get("url", "")
        base_score = float(item.get("score", 0.0))
        weighted_score = _clamp_score(base_score * _source_weight(url, strategy))
        weighted.append({**item, "score": weighted_score})
    reranked = sorted(
        weighted, key=lambda result: result.get("score", 0.0), reverse=True
    )
    return reranked[: configurable.rerank_top_k]
