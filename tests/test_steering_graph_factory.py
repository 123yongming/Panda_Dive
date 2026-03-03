"""Tests for deep researcher graph factory with steering support."""

from langgraph.checkpoint.memory import InMemorySaver

from Panda_Dive import build_deep_researcher, deep_researcher


def test_build_deep_researcher_returns_compiled_graph():
    """Factory should return a compiled graph with async entrypoints."""
    graph = build_deep_researcher()

    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "astream_events")


def test_build_deep_researcher_accepts_checkpointer():
    """Factory should wire the provided checkpointer."""
    saver = InMemorySaver()
    graph = build_deep_researcher(checkpointer=saver)

    assert graph.checkpointer is saver


def test_default_export_graph_still_available():
    """Existing deep_researcher export should remain compatible."""
    assert hasattr(deep_researcher, "ainvoke")
