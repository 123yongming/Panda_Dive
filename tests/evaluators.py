"""Evaluation functions for assessing research report quality."""

import json
import logging
import os
import re
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from Panda_Dive.configuration import Configuration
from Panda_Dive.utils import (
    create_chat_model,
    get_api_key_for_model,
    get_today_str,
    supports_structured_output,
)
from tests.prompts import (
    COMPLETENESS_PROMPT,
    CORRECTNESS_PROMPT,
    GROUNDEDNESS_PROMPT,
    OVERALL_QUALITY_PROMPT,
    RELEVANCE_PROMPT,
    STRUCTURE_PROMPT,
)

load_dotenv()
os.environ.setdefault("GET_API_KEYS_FROM_CONFIG", "false")


def _format_input_query(inputs: dict) -> str:
    messages = inputs.get("messages", [])
    if len(messages) == 1:
        return messages[0]["content"]

    role_to_string_format_map = {
        "user": "<user_input>\n{content}\n</user_input>",
        "assistant": "<assistant_follow_up>\n{content}\n</assistant_follow_up>",
    }

    return "\n\n".join(
        [
            role_to_string_format_map[message["role"]].format(
                content=message["content"]
            )
            for message in messages
        ]
    )


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for i in range(len(text)):
        if text[i] == "{":
            for j in range(len(text) - 1, i - 1, -1):
                if text[j] == "}":
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        pass
    return None


def _coerce_score(score: int | str | dict[str, Any]) -> int | str:
    if isinstance(score, int):
        return score
    if isinstance(score, str):
        try:
            return int(score)
        except ValueError:
            return score
    if isinstance(score, dict):
        return str(score)
    return str(score)


def _get_eval_model_settings() -> tuple[str, int, dict[str, dict]]:
    config = Configuration.from_runnable_config(None)
    model_name = os.environ.get("EVAL_MODEL") or config.final_report_model or "gpt-4o"
    model_max_tokens = int(
        os.environ.get(
            "EVAL_MODEL_MAX_TOKENS",
            config.final_report_model_max_tokens or 8192,
        )
    )
    runnable_config = {"configurable": config.model_dump()}
    return model_name, model_max_tokens, runnable_config


def _invoke_text_response(messages: list[dict], retries: int = 3) -> str:
    model_name, model_max_tokens, runnable_config = _get_eval_model_settings()
    base_model = create_chat_model(
        model_name=model_name,
        max_tokens=model_max_tokens,
        api_key=get_api_key_for_model(model_name, runnable_config),
    )
    model = base_model.with_retry(stop_after_attempt=retries)
    result = model.invoke(messages)
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _parse_score_from_text(text: str) -> int | None:
    if not text:
        return None
    result_json = _extract_json_from_text(text)
    if isinstance(result_json, dict) and "score" in result_json:
        try:
            return int(result_json["score"])
        except (TypeError, ValueError):
            return None
    patterns = [
        r"score\s*[:=]\s*([1-5])",
        r"\b([1-5])\s*/\s*5\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _invoke_structured(model_class: type, messages: list[dict], retries: int = 3):
    """Call model and parse structured output with retries."""
    model_name, model_max_tokens, runnable_config = _get_eval_model_settings()

    base_model = create_chat_model(
        model_name=model_name,
        max_tokens=model_max_tokens,
        api_key=get_api_key_for_model(model_name, runnable_config),
    )

    if supports_structured_output(model_name):
        model = base_model.with_structured_output(model_class).with_retry(
            stop_after_attempt=retries
        )
    else:
        model = base_model.with_retry(stop_after_attempt=retries)

    try:
        result = model.invoke(messages)
        if not isinstance(result, model_class):
            result_json = _extract_json_from_text(
                result.content if hasattr(result, "content") else str(result)
            )
            if result_json is not None:
                result = model_class.model_validate(result_json)
            else:
                raise ValueError(f"Could not parse result as {model_class.__name__}")
        return result
    except (ValidationError, json.JSONDecodeError, Exception) as e:
        logging.warning(f"Structured output failed: {e}")
        raise


class OverallQualityScore(BaseModel):
    """Evaluate the overall quality of a research report based on specific criteria."""

    research_depth: int | dict[str, Any] = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    source_quality: int | dict[str, Any] = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    critical_analysis: int | dict[str, Any] = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    balance_and_objectivity: int | dict[str, Any] = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )
    writing_quality: int | dict[str, Any] = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria)."
    )


def eval_overall_quality(inputs: dict, outputs: dict):
    """Evaluate the overall quality of a research report.

    Args:
        inputs: Input dictionary containing user messages.
        outputs: Output dictionary containing the final report.

    Returns:
        Dictionary containing evaluation score and feedback.
    """
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")
    user_input_content = (
        f"User input: {query}\n\nReport: \n\n{final_report}\n\n"
        "Evaluate whether the report meets the criteria and provide detailed justification for your evaluation."
    )
    messages = [
        {
            "role": "system",
            "content": OVERALL_QUALITY_PROMPT.format(today=get_today_str()),
        },
        {"role": "user", "content": user_input_content},
    ]
    try:
        eval_result = cast(
            OverallQualityScore, _invoke_structured(OverallQualityScore, messages)
        )
        return {
            "key": "overall_quality_score",
            "score": _coerce_score(eval_result.research_depth),
            "metadata": {
                "source_quality": _coerce_score(eval_result.source_quality),
                "critical_analysis": _coerce_score(eval_result.critical_analysis),
                "balance_and_objectivity": _coerce_score(
                    eval_result.balance_and_objectivity
                ),
                "writing_quality": _coerce_score(eval_result.writing_quality),
            },
        }
    except Exception as e:
        logging.warning(f"Overall quality evaluation failed: {e}")
        return {"key": "overall_quality_score", "score": None, "error": str(e)}


class RelevanceScore(BaseModel):
    """Evaluate the relevance of a research report based on specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for relevance (1 = doesn't meet at all, 5 = meets all criteria)."
    )


def eval_relevance(inputs: dict, outputs: dict):
    """Evaluate the relevance of a research report to the user's query.

    Args:
        inputs: Input dictionary containing user messages.
        outputs: Output dictionary containing the final report.

    Returns:
        Dictionary containing relevance score and feedback.
    """
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")
    user_input_content = (
        f"User input: {query}\n\nReport: \n\n{final_report}\n\n"
        "Evaluate whether the report meets the criteria and provide detailed justification for your evaluation."
    )
    messages = [
        {"role": "system", "content": RELEVANCE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content},
    ]
    try:
        eval_result = cast(RelevanceScore, _invoke_structured(RelevanceScore, messages))
        score = _coerce_score(eval_result.score)
        return {
            "key": "relevance_score",
            "score": score,
            "reasoning": eval_result.reasoning,
        }
    except Exception as e:
        logging.warning(f"Relevance evaluation failed: {e}")
        return {"key": "relevance_score", "score": None, "error": str(e)}


class StructureScore(BaseModel):
    """Evaluate the structure of a research report based on specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for structure and flow (1 = doesn't meet at all, 5 = meets all criteria)."
    )


def eval_structure(inputs: dict, outputs: dict):
    """Evaluate the structure and organization of a research report.

    Args:
        inputs: Input dictionary containing user messages.
        outputs: Output dictionary containing the final report.

    Returns:
        Dictionary containing structure score and feedback.
    """
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")
    user_input_content = STRUCTURE_PROMPT.format(
        user_question=query, report=final_report, today=get_today_str()
    )
    messages = [
        {"role": "system", "content": CORRECTNESS_PROMPT},
        {"role": "user", "content": user_input_content},
    ]
    try:
        eval_result = cast(StructureScore, _invoke_structured(StructureScore, messages))
        score = _coerce_score(eval_result.score)
        return {
            "key": "structure_and_cohesiveness_score",
            "score": score,
            "reasoning": eval_result.reasoning,
        }
    except Exception as e:
        logging.warning(f"Structure evaluation failed: {e}")
        return {
            "key": "structure_and_cohesiveness_score",
            "score": None,
            "error": str(e),
        }


class CorrectnessScore(BaseModel):
    """Evaluate the correctness of a research report based on specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for correctness (1 = doesn't meet at all, 5 = meets all criteria)."
    )


def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate the factual correctness of a research report.

    Args:
        inputs: Input dictionary containing user messages.
        outputs: Output dictionary containing the final report.
        reference_outputs: Reference output dictionary containing the authority answer.

    Returns:
        Dictionary containing correctness score and feedback.
    """
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")
    answer = reference_outputs.get("answer", "")
    user_input_content = (
        f"User input: {query}\n\nReport: \n\n{final_report}\n\n"
        f"Authority answer: \n\n{answer}\n\n"
        "Evaluate whether the report meets the criteria and provide detailed justification for your evaluation."
    )
    messages = [{"role": "user", "content": user_input_content}]
    try:
        eval_result = cast(
            CorrectnessScore, _invoke_structured(CorrectnessScore, messages)
        )
        score = _coerce_score(eval_result.score)
        return {
            "key": "correctness_score",
            "score": score,
            "reasoning": eval_result.reasoning,
        }
    except Exception as e:
        logging.warning(f"Correctness evaluation failed: {e}")
        try:
            fallback_text = _invoke_text_response(messages)
            fallback_score = _parse_score_from_text(fallback_text)
            if fallback_score is not None:
                return {
                    "key": "correctness_score",
                    "score": fallback_score,
                    "reasoning": fallback_text,
                }
        except Exception as fallback_error:
            logging.warning(f"Correctness fallback parsing failed: {fallback_error}")
        return {"key": "correctness_score", "score": None, "error": str(e)}


class GroundednessClaim(BaseModel):
    """A claim from the report, and whether it has context to support it."""

    text: str = Field(description="The claim extracted from the report.")
    is_grounded: bool = Field(
        description="Whether the claim is grounded in the context."
    )


class GroundednessScore(BaseModel):
    """Extract the claims and whether they are grounded in the context."""

    claims: list[GroundednessClaim] = Field(
        description="All claims extracted from the report, and whether or not they are grounded in the context."
    )


def eval_groundedness(inputs: dict, outputs: dict):
    """Evaluate the groundedness of claims in a research report.

    Args:
        inputs: Input dictionary containing user messages.
        outputs: Output dictionary containing the final report and raw notes.

    Returns:
        Dictionary containing groundedness evaluation results.
    """
    final_report = outputs.get("final_report", "")
    context = str(outputs.get("raw_notes", ""))
    user_input_content = GROUNDEDNESS_PROMPT.format(
        context=context, report=final_report, today=get_today_str()
    )
    messages = [{"role": "user", "content": user_input_content}]
    try:
        eval_result = cast(
            GroundednessScore, _invoke_structured(GroundednessScore, messages)
        )
    except Exception as exc:
        logging.warning(f"Groundedness evaluation failed: {exc}")
        return {"key": "groundedness_score", "score": None, "error": str(exc)}

    grounded_count = sum(1 for c in eval_result.claims if c.is_grounded)
    total_claims = len(eval_result.claims)
    score = (grounded_count / total_claims * 5) if total_claims > 0 else 0

    return {
        "key": "groundedness_score",
        "score": score,
        "grounded_count": grounded_count,
        "total_claims": total_claims,
    }


class CompletenessScore(BaseModel):
    """Evaluate the completeness of a research report based on specific criteria."""

    reasoning: str = Field(
        description="The reason for the score, including specific examples from the report."
    )
    score: int | str | dict[str, Any] = Field(
        description="Integer score 1-5 showing whether the report meets the provided criteria for completeness (1 = doesn't meet at all, 5 = meets all criteria)."
    )


def eval_completeness(inputs: dict, outputs: dict):
    """Evaluate the completeness of a research report.

    Args:
        inputs: Input dictionary containing user messages.
        outputs: Output dictionary containing the final report.

    Returns:
        Dictionary containing completeness score and feedback.
    """
    query = _format_input_query(inputs)
    final_report = outputs.get("final_report", "")
    research_brief = outputs.get("research_brief", "")
    user_input_content = COMPLETENESS_PROMPT.format(
        user_question=query,
        research_brief=research_brief,
        report=final_report,
        today=get_today_str(),
    )
    messages = [{"role": "user", "content": user_input_content}]
    try:
        eval_result = cast(
            CompletenessScore, _invoke_structured(CompletenessScore, messages)
        )
        score = _coerce_score(eval_result.score)
        return {
            "key": "completeness_score",
            "score": score,
            "reasoning": eval_result.reasoning,
        }
    except Exception as e:
        logging.warning(f"Completeness evaluation failed: {e}")
        return {"key": "completeness_score", "score": None, "error": str(e)}
