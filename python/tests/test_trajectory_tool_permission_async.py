from agentevals.trajectory.tool_permission import (
    create_async_trajectory_tool_permission_evaluator,
)

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
async def test_async_all_authorized():
    evaluator = create_async_trajectory_tool_permission_evaluator(
        allowed_tools=["search_kb"]
    )
    result = await evaluator(outputs=[_assistant_with_tools("search_kb")])
    assert result["score"] == 1.0


@pytest.mark.langsmith
async def test_async_unauthorized():
    evaluator = create_async_trajectory_tool_permission_evaluator(
        allowed_tools=["search_kb"]
    )
    result = await evaluator(outputs=[_assistant_with_tools("search_kb", "wipe_db")])
    assert result["score"] == 0.5
