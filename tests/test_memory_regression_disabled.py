"""Regression tests for memory defaults."""

from Panda_Dive.configuration import Configuration


def test_memory_disabled_by_default():
    """Memory feature flag should remain disabled unless explicitly enabled."""
    cfg = Configuration()
    assert cfg.memory_enabled is False
    assert cfg.memory_namespace_template == "memory.owner.{owner}"
    assert "memory_write_mode" not in Configuration.model_fields
