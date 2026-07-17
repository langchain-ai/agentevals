"""
Tests verifying that trajectory match scorers return diagnostic comments on failure.
"""
import json
import pytest

from agentevals.trajectory.match import create_trajectory_match_evaluator
from agentevals.types import ChatCompletionMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role, *, content=None, tool_calls=None):
    m = ChatCompletionMessage(role=role, content=content or "")
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


def _call(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


# ---------------------------------------------------------------------------
# strict
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_strict_pass_has_no_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    outputs = [
        _msg("user", content="hi"),
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
    ]
    result = evaluator(outputs=outputs, reference_outputs=outputs)
    assert result["score"] == True
    assert result["comment"] is None


@pytest.mark.langsmith
def test_strict_length_mismatch_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    outputs = [_msg("user", content="hi")]
    reference = [_msg("user", content="hi"), _msg("assistant", content="hello")]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "length" in result["comment"].lower()


@pytest.mark.langsmith
def test_strict_role_mismatch_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    outputs = [_msg("assistant", content="hi")]
    reference = [_msg("user", content="hi")]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "role" in result["comment"].lower()


@pytest.mark.langsmith
def test_strict_tool_name_mismatch_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    outputs = [_msg("assistant", tool_calls=[_call("search_web", {"q": "cats"})])]
    reference = [_msg("assistant", tool_calls=[_call("search_db", {"q": "cats"})])]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "search_web" in result["comment"] or "search_db" in result["comment"]


@pytest.mark.langsmith
def test_strict_tool_args_mismatch_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    outputs = [_msg("assistant", tool_calls=[_call("search", {"q": "dogs"})])]
    reference = [_msg("assistant", tool_calls=[_call("search", {"q": "cats"})])]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "search" in result["comment"]
    assert "mismatch" in result["comment"].lower() or "argument" in result["comment"].lower()


@pytest.mark.langsmith
def test_strict_tool_count_mismatch_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    outputs = [_msg("assistant", tool_calls=[
        _call("search", {"q": "cats"}),
        _call("search", {"q": "dogs"}),
    ])]
    reference = [_msg("assistant", tool_calls=[_call("search", {"q": "cats"})])]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None


# ---------------------------------------------------------------------------
# unordered
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_unordered_pass_has_no_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="unordered")
    outputs = [
        _msg("user", content="hi"),
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("lookup", {"id": 1})]),
    ]
    reference = [
        _msg("user", content="hi"),
        _msg("assistant", tool_calls=[_call("lookup", {"id": 1})]),
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == True
    assert result["comment"] is None


@pytest.mark.langsmith
def test_unordered_missing_tool_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="unordered")
    outputs = [_msg("assistant", tool_calls=[_call("search", {"q": "cats"})])]
    reference = [
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("lookup", {"id": 1})]),
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "missing" in result["comment"].lower()


@pytest.mark.langsmith
def test_unordered_extra_tool_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="unordered")
    outputs = [
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("unexpected_tool", {})]),
    ]
    reference = [_msg("assistant", tool_calls=[_call("search", {"q": "cats"})])]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "extra" in result["comment"].lower()


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_subset_pass_has_no_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="subset")
    outputs = [_msg("assistant", tool_calls=[_call("search", {"q": "cats"})])]
    reference = [
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("lookup", {"id": 1})]),
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == True
    assert result["comment"] is None


@pytest.mark.langsmith
def test_subset_extra_tool_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="subset")
    outputs = [
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("rogue_tool", {})]),
    ]
    reference = [_msg("assistant", tool_calls=[_call("search", {"q": "cats"})])]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "rogue_tool" in result["comment"]


# ---------------------------------------------------------------------------
# superset
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_superset_pass_has_no_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="superset")
    outputs = [
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("lookup", {"id": 1})]),
        _msg("assistant", tool_calls=[_call("extra_tool", {})]),
    ]
    reference = [
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("lookup", {"id": 1})]),
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == True
    assert result["comment"] is None


@pytest.mark.langsmith
def test_superset_missing_tool_comment():
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="superset")
    outputs = [_msg("assistant", tool_calls=[_call("search", {"q": "cats"})])]
    reference = [
        _msg("assistant", tool_calls=[_call("search", {"q": "cats"})]),
        _msg("assistant", tool_calls=[_call("required_tool", {"x": 1})]),
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference)
    assert result["score"] == False
    assert result["comment"] is not None
    assert "required_tool" in result["comment"]
    assert "missing" in result["comment"].lower()
