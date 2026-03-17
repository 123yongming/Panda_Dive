"""Tests for memory prompt integration helpers."""

from langchain_core.messages import HumanMessage, SystemMessage

from Panda_Dive.deepresearcher import build_supervisor_messages_with_memory


def test_build_supervisor_messages_with_memory_does_not_mutate_input():
    """Helper should return a new message list and preserve caller input."""
    original = [SystemMessage(content="Base prompt"), HumanMessage(content="brief")]
    updated = build_supervisor_messages_with_memory(original, "<memory_context>test</memory_context>")

    assert len(updated) == 2
    assert "<memory_context>test</memory_context>" in str(updated[0].content)
    assert "<memory_context>test</memory_context>" not in str(original[0].content)
