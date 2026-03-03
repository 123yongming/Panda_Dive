"""Tests for state reducers."""

from Panda_Dive.state import override_reducer


def test_override_reducer_override_payload_replaces_value():
    current_value = ["a", "b"]
    new_value = {"type": "override", "value": ["x"]}
    assert override_reducer(current_value, new_value) == ["x"]


def test_override_reducer_default_path_appends_values():
    current_value = ["a"]
    new_value = ["b", "c"]
    assert override_reducer(current_value, new_value) == ["a", "b", "c"]

