from agentevals.trajectory.tool_permission import (
    create_trajectory_tool_permission_evaluator,
)

from langchain_core.messages import AIMessage, HumanMessage

import json
import pytest


def _assistant_with_tools(*names):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": name, "arguments": json.dumps({})}} for name in names
        ],
    }


@pytest.mark.langsmith
def test_all_calls_authorized():
    evaluator = create_trajectory_tool_permission_evaluator(
        allowed_tools=["search_kb", "reply"]
    )
    outputs = [
        {"role": "user", "content": "help"},
        _assistant_with_tools("search_kb", "reply"),
    ]
    assert evaluator(outputs=outputs)["score"] == 1.0


@pytest.mark.langsmith
def test_unauthorized_tool():
    evaluator = create_trajectory_tool_permission_evaluator(allowed_tools=["search_kb"])
    outputs = [_assistant_with_tools("search_kb", "delete_account")]
    result = evaluator(outputs=outputs)
    assert result["score"] == 0.5


@pytest.mark.langsmith
def test_denied_tool_takes_precedence_over_allow():
    evaluator = create_trajectory_tool_permission_evaluator(
        allowed_tools=["search_kb", "wire_transfer"],
        denied_tools=["wire_transfer"],
    )
    outputs = [_assistant_with_tools("wire_transfer")]
    assert evaluator(outputs=outputs)["score"] == 0.0


@pytest.mark.langsmith
def test_no_tools_called_passes():
    evaluator = create_trajectory_tool_permission_evaluator(allowed_tools=["search_kb"])
    outputs = [{"role": "assistant", "content": "Here is your answer."}]
    assert evaluator(outputs=outputs)["score"] == 1.0


@pytest.mark.langsmith
def test_denylist_only():
    evaluator = create_trajectory_tool_permission_evaluator(denied_tools=["rm_rf"])
    outputs = [_assistant_with_tools("safe_tool", "rm_rf")]
    assert evaluator(outputs=outputs)["score"] == 0.5


@pytest.mark.langsmith
def test_langchain_messages():
    evaluator = create_trajectory_tool_permission_evaluator(
        allowed_tools=["get_weather"]
    )
    outputs = [
        HumanMessage(content="weather?"),
        AIMessage(
            content="",
            tool_calls=[{"id": "1", "name": "get_weather", "args": {"city": "SF"}}],
        ),
    ]
    assert evaluator(outputs=outputs)["score"] == 1.0


def test_requires_a_policy():
    with pytest.raises(ValueError):
        create_trajectory_tool_permission_evaluator()
