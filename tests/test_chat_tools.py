"""Tests for the chat tool declarations and their dispatch."""

import pytest

from context_fabric_mcp import chat
from context_fabric_mcp.cf_engine import CFEngine


@pytest.fixture(scope="module")
def engine():
    return CFEngine()


def _spec(name: str) -> dict:
    return next(
        t["function"] for t in chat.GENERAL_TOOLS if t["function"]["name"] == name
    )


@pytest.mark.parametrize("tool", ["search_constructions", "search_advanced"])
def test_search_tools_declare_scope_parameters(tool):
    props = _spec(tool)["parameters"]["properties"]
    for name in ("book", "chapter", "verse_start", "verse_end"):
        assert name in props
    assert _spec(tool)["parameters"]["required"] == ["template"]


def test_search_constructions_dispatch_passes_scope(engine: CFEngine):
    results = chat._execute_tool(
        engine,
        "search_constructions",
        {"template": "clause", "book": "PSA", "chapter": 23, "limit": 100},
    )
    assert len(results) == 17


def test_search_advanced_dispatch_passes_scope(engine: CFEngine):
    result = chat._execute_tool(
        engine,
        "search_advanced",
        {"template": "clause", "return_type": "count", "book": "PSA", "chapter": 23},
    )
    assert result["total_count"] == 17
