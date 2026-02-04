"""LLM-as-judge evaluators for DeepResearch evaluation.

This module implements 6 evaluators that use LLMs to judge research report quality:
- overall_quality: Multi-dimensional quality assessment
- relevance: Topic relevance and section relevance
- structure: Format and flow assessment
- correctness: Factual correctness against reference
- groundedness: Claims supported by context
- completeness: Coverage of all points
"""

import json
import logging
import os
import re
from typing import Any, cast

from langsmith.evaluation import run_evaluator
from langsmith.schemas import Example, Run
from prompts import (
    COMPLETENESS_PROMPT,
    CORRECTNESS_PROMPT,
    GROUNDEDNESS_PROMPT,
    OVERALL_QUALITY_PROMPT,
    RELEVANCE_PROMPT,
    STRUCTURE_PROMPT,
)
from pydantic import BaseModel, Field

from Panda_Dive.utils import (
    create_chat_model,
    get_today_str,
    supports_structured_output,
)

DEFAULT_EVAL_MODEL = "ark-code-latest"
DEFAULT_EVAL_MAX_TOKENS = 4096

logger = logging.getLogger(__name__)

_EVAL_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _extract_eval_payloads(
    run: Run, example: Example | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    inputs = run.inputs or {}
    if not inputs and example and example.inputs:
        inputs = example.inputs

    outputs = run.outputs or {}
    final_report = outputs.get("final_report") or outputs.get("output") or ""
    normalized_outputs = {
        "final_report": final_report,
        "raw_notes": outputs.get("raw_notes", ""),
        "research_brief": outputs.get("research_brief", ""),
    }

    reference_outputs = example.outputs if example and example.outputs else None
    return inputs, normalized_outputs, reference_outputs


def _get_evaluator_model(
    model_name: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Create and return the evaluator model instance.

    Args:
        model_name: Name of the model to use (default: gpt-4.1)
        api_key: API key for the model (optional)

    Returns:
        Configured chat model instance
    """
    model = model_name or DEFAULT_EVAL_MODEL
    key = api_key
    if key is None:
        lowered = model.lower()
        if lowered.startswith("ark"):
            key = os.getenv("ARK_API_KEY")
        elif lowered.startswith("deepseek"):
            key = os.getenv("DEEPSEEK_API_KEY")

    key = key or ""
    cache_key = (model, key)
    cached_model = _EVAL_MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model

    created_model = create_chat_model(
        model_name=model,
        max_tokens=DEFAULT_EVAL_MAX_TOKENS,
        api_key=key,
        tags=["langsmith:nostream"],
    )
    _EVAL_MODEL_CACHE[cache_key] = created_model
    return created_model


def _shorten_response_text(response_text: str, limit: int = 500) -> str:
    if len(response_text) <= limit:
        return response_text
    return f"{response_text[:limit]}...(truncated)"


def _append_json_instruction(content: str, instruction: str) -> str:
    return f"{content}\n\n{instruction}"


def _extract_json_block(response_text: str) -> str | None:
    text = response_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            return text[start : end + 1]
    return None


def _parse_json_response(response_text: str) -> dict[str, Any] | None:
    candidate = _extract_json_block(response_text) or response_text
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _format_input_query(inputs: dict[str, Any]) -> str:
    """Format the input messages into a query string.

    Args:
        inputs: Dictionary containing messages list

    Returns:
        Formatted query string
    """
    messages = inputs.get("messages", [])
    if len(messages) == 1:
        content = (
            messages[0].get("content", "")
            if isinstance(messages[0], dict)
            else str(messages[0])
        )
        return content

    role_to_string_format_map = {
        "user": "<user_input>\n{content}\n</user_input>",
        "assistant": "<assistant_follow_up>\n{content}\n</assistant_follow_up>",
    }

    formatted_messages = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = getattr(message, "type", "user")
            content = getattr(message, "content", str(message))

        format_template = role_to_string_format_map.get(
            role, role_to_string_format_map["user"]
        )
        formatted_messages.append(format_template.format(content=content))

    return "\n\n".join(formatted_messages)


class OverallQualityScore(BaseModel):
    """Score the overall quality of the report against specific criteria."""

    research_depth: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    source_quality: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    analytical_rigor: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    practical_value: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    balance_and_objectivity: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    writing_quality: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )


async def eval_overall_quality(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    model_name: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the overall quality of the research report.

    Assesses 6 dimensions: research depth, source quality, analytical rigor,
    practical value, balance/objectivity, and writing quality.

    Args:
        inputs: Dictionary containing the input messages
        outputs: Dictionary containing the final_report
        model_name: Optional model name override
        api_key: Optional API key

    Returns:
        List of score dictionaries with keys and normalized scores (0-1)
    """
    eval_model = _get_evaluator_model(model_name, api_key)
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")

    json_instruction = (
        "Respond ONLY with JSON: {"
        '"research_depth": 1-5, '
        '"source_quality": 1-5, '
        '"analytical_rigor": 1-5, '
        '"practical_value": 1-5, '
        '"balance_and_objectivity": 1-5, '
        '"writing_quality": 1-5'
        "}."
    )
    user_input_content = _append_json_instruction(
        f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation.""",
        json_instruction,
    )

    model_str = model_name or DEFAULT_EVAL_MODEL
    if not supports_structured_output(model_str):
        system_msg = {
            "role": "system",
            "content": OVERALL_QUALITY_PROMPT.format(today=get_today_str()),
        }
        user_msg = {"role": "user", "content": user_input_content}

        response = await eval_model.with_retry(stop_after_attempt=3).ainvoke(
            [system_msg, user_msg]
        )
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        parsed = _parse_json_response(response_text)
        try:
            if parsed is None:
                raise ValueError("No JSON object found")
            eval_result = OverallQualityScore(**parsed)
        except (ValueError, Exception) as e:
            logger.warning(
                "Failed to parse structured output: %s | model=%s | response=%r",
                e,
                model_str,
                _shorten_response_text(response_text),
            )
            return [
                {"key": "research_depth_score", "score": 0.5},
                {"key": "source_quality_score", "score": 0.5},
                {"key": "analytical_rigor_score", "score": 0.5},
                {"key": "practical_value_score", "score": 0.5},
                {"key": "balance_and_objectivity_score", "score": 0.5},
                {"key": "writing_quality_score", "score": 0.5},
            ]
    else:
        eval_result = cast(
            OverallQualityScore,
            eval_model.with_structured_output(OverallQualityScore)
            .with_retry(stop_after_attempt=3)
            .invoke(
                [
                    {
                        "role": "system",
                        "content": OVERALL_QUALITY_PROMPT.format(today=get_today_str()),
                    },
                    {"role": "user", "content": user_input_content},
                ]
            ),
        )

    return [
        {"key": "research_depth_score", "score": eval_result.research_depth / 5},
        {"key": "source_quality_score", "score": eval_result.source_quality / 5},
        {"key": "analytical_rigor_score", "score": eval_result.analytical_rigor / 5},
        {"key": "practical_value_score", "score": eval_result.practical_value / 5},
        {
            "key": "balance_and_objectivity_score",
            "score": eval_result.balance_and_objectivity / 5,
        },
        {"key": "writing_quality_score", "score": eval_result.writing_quality / 5},
    ]


class RelevanceScore(BaseModel):
    """Score the report relevance against specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for relevance (1 = doesn't meet at all, 5 = meets all criteria)."
    )


async def eval_relevance(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    model_name: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate the relevance of the research report to the user query.

    Assesses topic relevance, section relevance, citations, and overall quality.

    Args:
        inputs: Dictionary containing the input messages
        outputs: Dictionary containing the final_report
        model_name: Optional model name override
        api_key: Optional API key

    Returns:
        Dictionary with key, normalized score (0-1), and reasoning comment
    """
    eval_model = _get_evaluator_model(model_name, api_key)
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")

    json_instruction = 'Respond ONLY with JSON: {"reasoning": "string", "score": 1-5}.'
    user_input_content = _append_json_instruction(
        f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation.""",
        json_instruction,
    )

    model_str = model_name or DEFAULT_EVAL_MODEL
    if not supports_structured_output(model_str):
        system_msg = {
            "role": "system",
            "content": RELEVANCE_PROMPT.format(today=get_today_str()),
        }
        user_msg = {"role": "user", "content": user_input_content}

        response = await eval_model.with_retry(stop_after_attempt=3).ainvoke(
            [system_msg, user_msg]
        )
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        parsed = _parse_json_response(response_text)
        try:
            if parsed is None:
                raise ValueError("No JSON object found")
            eval_result = RelevanceScore(**parsed)
        except (ValueError, Exception) as e:
            logger.warning(
                "Failed to parse structured output: %s | model=%s | response=%r",
                e,
                model_str,
                _shorten_response_text(response_text),
            )
            return {
                "key": "relevance_score",
                "score": 0.5,
                "comment": "Failed to parse evaluator response",
            }
    else:
        eval_result = cast(
            RelevanceScore,
            eval_model.with_structured_output(RelevanceScore)
            .with_retry(stop_after_attempt=3)
            .invoke(
                [
                    {
                        "role": "system",
                        "content": RELEVANCE_PROMPT.format(today=get_today_str()),
                    },
                    {"role": "user", "content": user_input_content},
                ]
            ),
        )

    return {
        "key": "relevance_score",
        "score": eval_result.score / 5,
        "comment": eval_result.reasoning,
    }


class StructureScore(BaseModel):
    """Score the report structure against specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for structure and flow (1 = doesn't meet at all, 5 = meets all criteria)."
    )


async def eval_structure(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    model_name: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate the structure and flow of the research report.

    Assesses format appropriateness, logical flow, and use of structural elements.

    Args:
        inputs: Dictionary containing the input messages
        outputs: Dictionary containing the final_report
        model_name: Optional model name override
        api_key: Optional API key

    Returns:
        Dictionary with key, normalized score (0-1), and reasoning comment
    """
    eval_model = _get_evaluator_model(model_name, api_key)
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")

    json_instruction = 'Respond ONLY with JSON: {"reasoning": "string", "score": 1-5}.'
    user_input_content = _append_json_instruction(
        STRUCTURE_PROMPT.format(
            user_question=query,
            report=final_report,
            today=get_today_str(),
        ),
        json_instruction,
    )

    model_str = model_name or DEFAULT_EVAL_MODEL
    if not supports_structured_output(model_str):
        user_msg = {"role": "user", "content": user_input_content}

        response = await eval_model.with_retry(stop_after_attempt=3).ainvoke([user_msg])
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        parsed = _parse_json_response(response_text)
        try:
            if parsed is None:
                raise ValueError("No JSON object found")
            eval_result = StructureScore(**parsed)
        except (ValueError, Exception) as e:
            logger.warning(
                "Failed to parse structured output: %s | model=%s | response=%r",
                e,
                model_str,
                _shorten_response_text(response_text),
            )
            return {
                "key": "structure_and_cohesiveness_score",
                "score": 0.5,
                "comment": "Failed to parse evaluator response",
            }
    else:
        eval_result = cast(
            StructureScore,
            eval_model.with_structured_output(StructureScore)
            .with_retry(stop_after_attempt=3)
            .invoke(
                [
                    {"role": "user", "content": user_input_content},
                ]
            ),
        )

    return {
        "key": "structure_and_cohesiveness_score",
        "score": eval_result.score / 5,
        "comment": eval_result.reasoning,
    }


class CorrectnessScore(BaseModel):
    """Score the report correctness against specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for correctness (1 = doesn't meet at all, 5 = meets all criteria)."
    )


async def eval_correctness(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
    model_name: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate the factual correctness of the research report.

    Compares the report against a reference answer from an independent authority.

    Args:
        inputs: Dictionary containing the input messages
        outputs: Dictionary containing the final_report
        reference_outputs: Dictionary containing the reference answer
        model_name: Optional model name override
        api_key: Optional API key

    Returns:
        Dictionary with key, normalized score (0-1), and reasoning comment
    """
    eval_model = _get_evaluator_model(model_name, api_key)
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")
    answer = reference_outputs.get("answer", "")

    json_instruction = 'Respond ONLY with JSON: {"reasoning": "string", "score": 1-5}.'
    user_input_content = _append_json_instruction(
        CORRECTNESS_PROMPT.format(
            user_question=query,
            report=final_report,
            answer=answer,
            today=get_today_str(),
        ),
        json_instruction,
    )

    model_str = model_name or DEFAULT_EVAL_MODEL
    if not supports_structured_output(model_str):
        user_msg = {"role": "user", "content": user_input_content}

        response = await eval_model.with_retry(stop_after_attempt=3).ainvoke([user_msg])
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        parsed = _parse_json_response(response_text)
        try:
            if parsed is None:
                raise ValueError("No JSON object found")
            eval_result = CorrectnessScore(**parsed)
        except (ValueError, Exception) as e:
            logger.warning(
                "Failed to parse structured output: %s | model=%s | response=%r",
                e,
                model_str,
                _shorten_response_text(response_text),
            )
            return {
                "key": "correctness_score",
                "score": 0.5,
                "comment": "Failed to parse evaluator response",
            }
    else:
        eval_result = cast(
            CorrectnessScore,
            eval_model.with_structured_output(CorrectnessScore)
            .with_retry(stop_after_attempt=3)
            .invoke(
                [
                    {"role": "user", "content": user_input_content},
                ]
            ),
        )

    return {
        "key": "correctness_score",
        "score": eval_result.score / 5,
        "comment": eval_result.reasoning,
    }


class GroundednessClaim(BaseModel):
    """A claim from the report, and whether or not it is grounded in the context."""

    claim: str = Field(description="The claim extracted from the report.")
    grounded: bool = Field(description="Whether the claim is grounded in the context.")


class GroundednessScore(BaseModel):
    """Extract the claims and whether they are grounded in the context."""

    claims: list[GroundednessClaim] = Field(
        description="All claims extracted from the report, and whether or not they are grounded in the context."
    )


async def eval_groundedness(
    outputs: dict[str, Any],
    model_name: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate how well the research report is grounded in the retrieved context.

    Extracts claims from the report and checks if they are supported by the context.

    Args:
        outputs: Dictionary containing final_report and raw_notes
        model_name: Optional model name override
        api_key: Optional API key

    Returns:
        Dictionary with key, normalized score (0-1), and claims as comment
    """
    eval_model = _get_evaluator_model(model_name, api_key)
    final_report = outputs.get("final_report", "")
    context = str(outputs.get("raw_notes", ""))

    json_instruction = (
        'Respond ONLY with JSON: {"claims": ['
        '{"claim": "string", "grounded": true|false}]}'
    )
    user_input_content = _append_json_instruction(
        GROUNDEDNESS_PROMPT.format(
            context=context,
            report=final_report,
            today=get_today_str(),
        ),
        json_instruction,
    )

    model_str = model_name or DEFAULT_EVAL_MODEL
    if not supports_structured_output(model_str):
        user_msg = {"role": "user", "content": user_input_content}

        response = await eval_model.with_retry(stop_after_attempt=3).ainvoke([user_msg])
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        parsed = _parse_json_response(response_text)
        if isinstance(parsed, dict) and "claims" in parsed:
            normalized_claims = []
            for claim in (
                parsed.get("claims", [])
                if isinstance(parsed.get("claims"), list)
                else []
            ):
                if isinstance(claim, dict) and "claim" not in claim and "text" in claim:
                    claim = {**claim, "claim": claim.get("text")}
                normalized_claims.append(claim)
            parsed = {**parsed, "claims": normalized_claims}
        try:
            if parsed is None:
                raise ValueError("No JSON object found")
            eval_result = GroundednessScore(**parsed)
        except (ValueError, Exception) as e:
            logger.warning(
                "Failed to parse structured output: %s | model=%s | response=%r",
                e,
                model_str,
                _shorten_response_text(response_text),
            )
            return {
                "key": "groundedness_score",
                "score": 0.5,
                "comment": "Failed to parse evaluator response",
            }
    else:
        eval_result = cast(
            GroundednessScore,
            eval_model.with_structured_output(GroundednessScore)
            .with_retry(stop_after_attempt=3)
            .invoke(
                [
                    {"role": "user", "content": user_input_content},
                ]
            ),
        )

    claims = eval_result.claims
    if not claims:
        return {
            "key": "groundedness_score",
            "score": 0.0,
            "comment": "No claims found in report",
        }

    grounded_claims = [c for c in claims if c.grounded]
    score = len(grounded_claims) / len(claims)

    return {
        "key": "groundedness_score",
        "score": score,
        "comment": str([{"claim": c.claim, "grounded": c.grounded} for c in claims]),
    }


class CompletenessScore(BaseModel):
    """Score the report completeness against specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for completeness (1 = doesn't meet at all, 5 = meets all criteria)."
    )


async def eval_completeness(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    model_name: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate the completeness of the research report.

    Assesses whether the report answers all points from the user's question
    and fully encompasses the research brief.

    Args:
        inputs: Dictionary containing the input messages
        outputs: Dictionary containing the final_report and research_brief
        model_name: Optional model name override
        api_key: Optional API key

    Returns:
        Dictionary with key, normalized score (0-1), and reasoning comment
    """
    eval_model = _get_evaluator_model(model_name, api_key)
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")
    research_brief = outputs.get("research_brief", "")

    json_instruction = 'Respond ONLY with JSON: {"reasoning": "string", "score": 1-5}.'
    user_input_content = _append_json_instruction(
        COMPLETENESS_PROMPT.format(
            research_brief=research_brief,
            user_question=query,
            report=final_report,
            today=get_today_str(),
        ),
        json_instruction,
    )

    model_str = model_name or DEFAULT_EVAL_MODEL
    if not supports_structured_output(model_str):
        user_msg = {"role": "user", "content": user_input_content}

        response = await eval_model.with_retry(stop_after_attempt=3).ainvoke([user_msg])
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        parsed = _parse_json_response(response_text)
        try:
            if parsed is None:
                raise ValueError("No JSON object found")
            eval_result = CompletenessScore(**parsed)
        except (ValueError, Exception) as e:
            logger.warning(
                "Failed to parse structured output: %s | model=%s | response=%r",
                e,
                model_str,
                _shorten_response_text(response_text),
            )
            return {
                "key": "completeness_score",
                "score": 0.5,
                "comment": "Failed to parse evaluator response",
            }
    else:
        eval_result = cast(
            CompletenessScore,
            eval_model.with_structured_output(CompletenessScore)
            .with_retry(stop_after_attempt=3)
            .invoke(
                [
                    {"role": "user", "content": user_input_content},
                ]
            ),
        )

    return {
        "key": "completeness_score",
        "score": eval_result.score / 5,
        "comment": eval_result.reasoning,
    }


async def evaluate_report(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Run all evaluators on a research report.

    This convenience function runs all 6 evaluators (overall_quality, relevance,
    structure, correctness, groundedness, completeness) and returns a combined
    list of all scores.

    Args:
        inputs: Dictionary containing the input messages
        outputs: Dictionary containing final_report, raw_notes, and research_brief
        reference_outputs: Optional dictionary containing reference answer for correctness
        model_name: Optional model name override
        api_key: Optional API key

    Returns:
        Combined list of all evaluation scores from all evaluators
    """
    all_scores = []

    try:
        quality_scores = await eval_overall_quality(
            inputs, outputs, model_name, api_key
        )
        all_scores.extend(quality_scores)
    except Exception as e:
        logger.warning(f"Overall quality evaluation failed: {e}")
        all_scores.extend(
            [
                {
                    "key": "research_depth_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                },
                {
                    "key": "source_quality_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                },
                {
                    "key": "analytical_rigor_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                },
                {
                    "key": "practical_value_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                },
                {
                    "key": "balance_and_objectivity_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                },
                {
                    "key": "writing_quality_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                },
            ]
        )

    try:
        relevance_score = await eval_relevance(inputs, outputs, model_name, api_key)
        all_scores.append(relevance_score)
    except Exception as e:
        logger.warning(f"Relevance evaluation failed: {e}")
        all_scores.append(
            {
                "key": "relevance_score",
                "score": 0.0,
                "comment": f"Evaluation failed: {e}",
            }
        )

    try:
        structure_score = await eval_structure(inputs, outputs, model_name, api_key)
        all_scores.append(structure_score)
    except Exception as e:
        logger.warning(f"Structure evaluation failed: {e}")
        all_scores.append(
            {
                "key": "structure_and_cohesiveness_score",
                "score": 0.0,
                "comment": f"Evaluation failed: {e}",
            }
        )

    if reference_outputs:
        try:
            correctness_score = await eval_correctness(
                inputs, outputs, reference_outputs, model_name, api_key
            )
            all_scores.append(correctness_score)
        except Exception as e:
            logger.warning(f"Correctness evaluation failed: {e}")
            all_scores.append(
                {
                    "key": "correctness_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                }
            )

    if "raw_notes" in outputs:
        try:
            groundedness_score = await eval_groundedness(outputs, model_name, api_key)
            all_scores.append(groundedness_score)
        except Exception as e:
            logger.warning(f"Groundedness evaluation failed: {e}")
            all_scores.append(
                {
                    "key": "groundedness_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                }
            )

    if "research_brief" in outputs:
        try:
            completeness_score = await eval_completeness(
                inputs, outputs, model_name, api_key
            )
            all_scores.append(completeness_score)
        except Exception as e:
            logger.warning(f"Completeness evaluation failed: {e}")
            all_scores.append(
                {
                    "key": "completeness_score",
                    "score": 0.0,
                    "comment": f"Evaluation failed: {e}",
                }
            )

    return all_scores


@run_evaluator
async def overall_quality_evaluator(run: Run, example: Example | None = None):
    """Evaluate overall quality of research output.

    Args:
        run: LangSmith run object containing inputs and outputs.
        example: Example object containing reference data.

    Returns:
        Dictionary with evaluation score and feedback.
    """
    inputs, outputs, _ = _extract_eval_payloads(run, example)
    return await eval_overall_quality(inputs, outputs)


@run_evaluator
async def relevance_evaluator(run: Run, example: Example | None = None):
    """Evaluate relevance of research output to the original query.

    Args:
        run: LangSmith run object containing inputs and outputs.
        example: Example object containing reference data.

    Returns:
        Dictionary with evaluation score and feedback.
    """
    inputs, outputs, _ = _extract_eval_payloads(run, example)
    return await eval_relevance(inputs, outputs)


@run_evaluator
async def structure_evaluator(run: Run, example: Example | None = None):
    """Evaluate structural quality of research output.

    Args:
        run: LangSmith run object containing inputs and outputs.
        example: Example object containing reference data.

    Returns:
        Dictionary with evaluation score and feedback.
    """
    inputs, outputs, _ = _extract_eval_payloads(run, example)
    return await eval_structure(inputs, outputs)


@run_evaluator
async def correctness_evaluator(run: Run, example: Example | None = None):
    """Evaluate factual correctness of research output against reference.

    Args:
        run: LangSmith run object containing inputs and outputs.
        example: Example object containing reference data.

    Returns:
        Dictionary with evaluation score and feedback.
    """
    inputs, outputs, reference_outputs = _extract_eval_payloads(run, example)
    if not reference_outputs:
        return {
            "key": "correctness_score",
            "score": 0.0,
            "comment": "No reference output provided",
        }
    return await eval_correctness(inputs, outputs, reference_outputs)


@run_evaluator
async def groundedness_evaluator(run: Run, example: Example | None = None):
    """Evaluate factual groundedness of research output in search results.

    Args:
        run: LangSmith run object containing inputs and outputs.
        example: Example object containing reference data.

    Returns:
        Dictionary with evaluation score and feedback.
    """
    _, outputs, _ = _extract_eval_payloads(run, example)
    return await eval_groundedness(outputs)


@run_evaluator
async def completeness_evaluator(run: Run, example: Example | None = None):
    """Evaluate completeness of research output coverage.

    Args:
        run: LangSmith run object containing inputs and outputs.
        example: Example object containing reference data.

    Returns:
        Dictionary with evaluation score and feedback.
    """
    inputs, outputs, _ = _extract_eval_payloads(run, example)
    return await eval_completeness(inputs, outputs)
