"""Tests for the chat tool declarations and their dispatch."""

import json

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


class TestFitResult:
    def test_small_result_is_unchanged(self):
        out, data = chat._fit_result([{"a": 1}], 1000)
        assert json.loads(out) == data == [{"a": 1}]

    def test_long_list_is_cut_at_whole_items_and_flagged(self):
        items = [{"n": i, "text": "x" * 50} for i in range(100)]
        out, data = chat._fit_result(items, 1000)
        assert len(out) <= 1000
        assert data["truncated"] is True
        assert data["total"] == 100
        assert 0 < data["shown"] < 100
        assert data["results"] == items[: data["shown"]]  # a clean prefix
        assert "NOT complete" in data["note"]

    def test_results_list_inside_a_dict_keeps_the_other_keys(self):
        result = {"total_count": 500, "cursor": "abc", "results": [{"t": "y" * 40}] * 100}
        out, data = chat._fit_result(result, 1000)
        assert len(out) <= 1000
        assert data["total_count"] == 500 and data["cursor"] == "abc"
        assert data["truncated"] is True and data["total"] == 100

    def test_single_oversized_item_still_flags_truncation(self):
        out, data = chat._fit_result([{"text": "z" * 5000}], 1000)
        assert data["truncated"] is True
        assert "partial" in data

    def test_plain_dict_is_flagged(self):
        out, data = chat._fit_result({"text": "z" * 5000}, 1000)
        assert data["truncated"] is True and "NOT complete" in data["note"]
