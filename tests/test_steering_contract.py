"""Tests for steering configuration and state contracts."""

from Panda_Dive.configuration import Configuration
from Panda_Dive.state import SupervisorState


def test_configuration_steering_defaults():
    """Steering should be opt-in with explicit default commands."""
    config = Configuration()

    assert config.enable_steering is False
    assert config.steering_command_prefix == "/steer"
    assert config.steering_continue_command == "/continue"


def test_supervisor_state_has_steering_fields():
    """Supervisor state should expose steering-related channels."""
    annotations = SupervisorState.__annotations__

    assert "steering_history" in annotations
    assert "steering_last_command" in annotations
    assert "steering_warnings" in annotations
