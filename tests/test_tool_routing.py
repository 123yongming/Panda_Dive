"""Tests for search tool routing and tool assembly."""

import pytest

from Panda_Dive.configuration import SearchAPI
from Panda_Dive.utils import get_all_tools, get_search_tool


@pytest.mark.anyio
async def test_get_search_tool_routes_tavily():
    tools = await get_search_tool(SearchAPI.TAVILY)
    assert len(tools) == 1
    assert tools[0].name == "tavily_search"


@pytest.mark.anyio
async def test_get_search_tool_routes_duckduckgo():
    tools = await get_search_tool(SearchAPI.DUCKDUCKGO)
    assert len(tools) == 1
    assert tools[0].name == "duckduckgo_search"


@pytest.mark.anyio
async def test_get_search_tool_routes_none():
    tools = await get_search_tool(SearchAPI.NONE)
    assert tools == []


@pytest.mark.anyio
async def test_get_all_tools_without_search_and_mcp(monkeypatch):
    async def _fake_load_mcp_tools(config, existing_tool_names):
        return []

    monkeypatch.setattr("Panda_Dive.utils.load_mcp_tools", _fake_load_mcp_tools)
    config = {"configurable": {"search_api": "none"}}
    tools = await get_all_tools(config)
    names = [tool.name for tool in tools]

    assert "think_tool" in names
    assert "ResearchComplete" in names
    assert "tavily_search" not in names
    assert "duckduckgo_search" not in names

