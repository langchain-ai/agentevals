from typing import Optional, Union

from agentevals.trajectory.utils import (
    _normalize_to_openai_messages_list,
    _extract_tool_calls,
)
from agentevals.types import (
    ChatCompletionMessage,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
)
from agentevals.utils import _run_evaluator, _arun_evaluator

from langchain_core.messages import BaseMessage


def _score_tool_permissions(
    outputs: list[ChatCompletionMessage],
    allowed_tools: Optional[set[str]],
    denied_tools: set[str],
) -> float:
    tool_calls = _extract_tool_calls(outputs)
    if not tool_calls:
        # No tools were called, so no permission boundary could be violated.
        return 1.0
    unauthorized = 0
    for call in tool_calls:
        name = call.get("name")
        if name in denied_tools:
            unauthorized += 1
        elif allowed_tools is not None and name not in allowed_tools:
            unauthorized += 1
    total = len(tool_calls)
    return (total - unauthorized) / total


def _make_scorer(allowed_tools: Optional[set[str]], denied_tools: set[str]):
    def _scorer(*, outputs: list[ChatCompletionMessage], **kwargs) -> float:
        return _score_tool_permissions(outputs, allowed_tools, denied_tools)

    return _scorer


def create_trajectory_tool_permission_evaluator(
    *,
    allowed_tools: Optional[list[str]] = None,
    denied_tools: Optional[list[str]] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that checks whether an agent only called tools it was
    authorized to, based on a permission policy (independent of any reference trajectory).

    Unlike ``create_trajectory_match_evaluator``, which compares called tools against a
    reference trajectory, this evaluator enforces least privilege: it flags any tool call
    outside the granted policy, regardless of whether the task was completed.

    Args:
        allowed_tools (Optional[list[str]]): Allowlist of permitted tool names. If provided,
            any called tool not in this list counts as unauthorized (least privilege).
        denied_tools (Optional[list[str]]): Denylist of forbidden tool names. Any called tool
            in this list counts as unauthorized. A denial always takes precedence over an allow.

    Returns:
        SimpleEvaluator: A function that accepts `outputs` (the agent trajectory) and returns
        an EvaluatorResult whose score is the fraction of tool calls that were authorized
        (1.0 when no tools were called).

    Example:
    ```python
    evaluator = create_trajectory_tool_permission_evaluator(
        allowed_tools=["search_kb", "reply_to_customer"],
        denied_tools=["issue_refund"],
    )
    result = evaluator(outputs=...)
    ```
    """
    if allowed_tools is None and denied_tools is None:
        raise ValueError(
            "create_trajectory_tool_permission_evaluator requires at least one of "
            "`allowed_tools` (an allowlist) or `denied_tools` (a denylist)."
        )
    allowed = set(allowed_tools) if allowed_tools is not None else None
    denied = set(denied_tools or [])
    scorer = _make_scorer(allowed, denied)

    def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ):
        outputs = _normalize_to_openai_messages_list(outputs)
        return _run_evaluator(
            run_name="trajectory_tool_permission",
            scorer=scorer,
            feedback_key="trajectory_tool_permission",
            outputs=outputs,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_trajectory_tool_permission_evaluator(
    *,
    allowed_tools: Optional[list[str]] = None,
    denied_tools: Optional[list[str]] = None,
) -> SimpleAsyncEvaluator:
    """Async version of ``create_trajectory_tool_permission_evaluator``."""
    if allowed_tools is None and denied_tools is None:
        raise ValueError(
            "create_async_trajectory_tool_permission_evaluator requires at least one of "
            "`allowed_tools` (an allowlist) or `denied_tools` (a denylist)."
        )
    allowed = set(allowed_tools) if allowed_tools is not None else None
    denied = set(denied_tools or [])
    scorer = _make_scorer(allowed, denied)

    async def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ):
        outputs = _normalize_to_openai_messages_list(outputs)
        return await _arun_evaluator(
            run_name="trajectory_tool_permission",
            scorer=scorer,
            feedback_key="trajectory_tool_permission",
            outputs=outputs,
            **kwargs,
        )

    return _wrapped_evaluator
