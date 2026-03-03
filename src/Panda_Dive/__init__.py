"""Panda_Dive - 领域深度搜索工具。."""

from .configuration import Configuration
from .deepresearcher import build_deep_researcher, deep_researcher

__version__ = "3.0.0"
__all__ = ["Configuration", "deep_researcher", "build_deep_researcher"]
