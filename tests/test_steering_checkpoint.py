"""Tests for steering checkpoint node routing and updates."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from Panda_Dive.configuration import Configuration
from Panda_Dive.deepresearcher import steering_checkpoint


@pytest.mark.anyio
async def test_steering_checkpoint_disabled_skips_interrupt(monkeypatch):
    """When steering is disabled, checkpoint should no-op to supervisor."""
    called = {"value": False}

    def _interrupt(_value):
        called["value"] = True
        return "/continue"

    monkeypatch.setattr("Panda_Dive.deepresearcher.interrupt", _interrupt)

    state = {
        "research_brief": "Initial brief",
        "research_iterations": 1,
        "notes": [],
        "supervisor_messages": [],
    }
    cfg = {"configurable": Configuration(enable_steering=False).model_dump()}

    result = await steering_checkpoint(state, cfg)

    assert result.goto == "supervisor"
    assert called["value"] is False


@pytest.mark.anyio
async def test_steering_checkpoint_continue_command(monkeypatch):
    """Continue command should not modify the research brief."""
    monkeypatch.setattr(
        "Panda_Dive.deepresearcher.interrupt",
        lambda _value: "/continue",
    )

    state = {
        "research_brief": "Initial brief",
        "research_iterations": 2,
        "notes": [],
        "supervisor_messages": [],
    }
    cfg = {"configurable": Configuration(enable_steering=True).model_dump()}

    result = await steering_checkpoint(state, cfg)

    assert result.goto == "supervisor"
    assert result.update["steering_last_command"] == "/continue"
    assert "research_brief" not in result.update


@pytest.mark.anyio
async def test_steering_checkpoint_steer_command_updates_brief(monkeypatch):
    """Steer command should update brief and replace old brief message."""
    directive = "Focus on papers published after 2024."
    monkeypatch.setattr(
        "Panda_Dive.deepresearcher.interrupt",
        lambda _value: f"/steer {directive}",
    )

    old_brief = "Initial brief"
    state = {
        "research_brief": old_brief,
        "research_iterations": 3,
        "notes": [],
        "supervisor_messages": [
            SystemMessage(content="system prompt"),
            HumanMessage(content=old_brief),
            AIMessage(content="tool planning"),
        ],
    }
    cfg = {"configurable": Configuration(enable_steering=True).model_dump()}

    result = await steering_checkpoint(state, cfg)

    assert result.goto == "supervisor"
    updated_brief = result.update["research_brief"]
    assert directive in updated_brief
    assert result.update["steering_history"] == [directive]
    assert result.update["supervisor_messages"]["type"] == "override"
    updated_messages = result.update["supervisor_messages"]["value"]
    contents = [str(getattr(message, "content", "")) for message in updated_messages]
    assert old_brief not in contents
    assert updated_brief in contents


@pytest.mark.anyio
async def test_steering_checkpoint_rebuilds_context_when_brief_missing(monkeypatch):
    """Steer command should rebuild minimal context if old brief is absent."""
    directive = "Use only official sources."
    monkeypatch.setattr(
        "Panda_Dive.deepresearcher.interrupt",
        lambda _value: f"/steer {directive}",
    )

    state = {
        "research_brief": "Original brief",
        "research_iterations": 4,
        "notes": [],
        "supervisor_messages": [AIMessage(content="no human brief in this context")],
    }
    cfg = {"configurable": Configuration(enable_steering=True).model_dump()}

    result = await steering_checkpoint(state, cfg)

    assert result.goto == "supervisor"
    assert result.update["supervisor_messages"]["type"] == "override"
    rebuilt_messages = result.update["supervisor_messages"]["value"]
    assert len(rebuilt_messages) == 2
    assert isinstance(rebuilt_messages[0], SystemMessage)
    assert isinstance(rebuilt_messages[1], HumanMessage)
    assert directive in rebuilt_messages[1].content
    warnings = result.update.get("steering_warnings", [])
    assert any("fallback" in warning.lower() for warning in warnings)
