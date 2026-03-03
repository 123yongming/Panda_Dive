"""Conventions for model initialization entry points."""

from pathlib import Path


def test_deepresearcher_does_not_import_init_chat_model_directly():
    source = Path("src/Panda_Dive/deepresearcher.py").read_text(encoding="utf-8")
    assert "from langchain.chat_models import init_chat_model" not in source
    assert "configurable_model = init_chat_model(" not in source

