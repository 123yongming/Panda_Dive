"""Tests for steering command parsing helpers."""

from Panda_Dive.deepresearcher import (
    _apply_steering_directive_to_brief,
    _parse_steering_command,
)


def test_parse_continue_command():
    """Continue command should pass through without warning."""
    parsed = _parse_steering_command(
        "/continue",
        steering_command_prefix="/steer",
        steering_continue_command="/continue",
    )

    assert parsed["action"] == "continue"
    assert parsed["directive"] is None
    assert parsed["warning"] is None


def test_parse_steering_command_with_directive():
    """Steering command should extract the natural-language directive."""
    parsed = _parse_steering_command(
        "/steer focus on peer-reviewed sources",
        steering_command_prefix="/steer",
        steering_continue_command="/continue",
    )

    assert parsed["action"] == "steer"
    assert parsed["directive"] == "focus on peer-reviewed sources"
    assert parsed["warning"] is None


def test_parse_invalid_command_falls_back_to_continue():
    """Invalid command should continue and emit a warning."""
    parsed = _parse_steering_command(
        "/unknown do something",
        steering_command_prefix="/steer",
        steering_continue_command="/continue",
    )

    assert parsed["action"] == "continue"
    assert parsed["directive"] is None
    assert parsed["warning"] is not None


def test_apply_steering_directive_to_brief():
    """Steering directive should be appended deterministically to brief."""
    brief = "Research the market landscape for AI coding tools."
    directive = "Prioritize peer-reviewed sources from 2024 onwards."

    updated = _apply_steering_directive_to_brief(brief, directive)

    assert "Research the market landscape for AI coding tools." in updated
    assert "Prioritize peer-reviewed sources from 2024 onwards." in updated
