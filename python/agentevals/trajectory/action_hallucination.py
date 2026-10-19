from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from agentevals.trajectory._action_claims import (
    Claim,
    ToolSpec,
    extract_claims,
    family_ran,
    normalize_tool_specs,
)
from agentevals.trajectory.utils import _normalize_to_openai_messages_list
from agentevals.types import (
    ChatCompletionMessage,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
)
from agentevals.utils import _run_evaluator, _arun_evaluator

from langchain_core.messages import BaseMessage, ToolMessage


def _failed_tool_call_ids(
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
) -> list[str]:
    # Read before normalizing: the OpenAI-format conversion drops ToolMessage.status.
    messages: Any = outputs
    if isinstance(messages, dict):
        if "role" in messages:
            messages = [messages]
        elif "messages" in messages:
            messages = messages["messages"]
    if not isinstance(messages, list):
        messages = [messages]
    failed = []
    status: Optional[str]
    tool_call_id: Optional[str]
    for message in messages:
        if isinstance(message, ToolMessage):
            status, tool_call_id = message.status, message.tool_call_id
        elif isinstance(message, dict) and message.get("role") == "tool":
            status, tool_call_id = message.get("status"), message.get("tool_call_id")
        else:
            continue
        # A failure without an id cannot be paired with its call, so it is ignored.
        if status == "error" and tool_call_id:
            failed.append(tool_call_id)
    return failed


def _final_reply(messages: list[ChatCompletionMessage]) -> str:
    for message in reversed(messages):
        if message["role"] == "assistant":
            content = message.get("content") or ""
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return content
    return ""


def _find_fabricated(
    claims: Sequence[Claim], ok_tools: set[str], error_tools: set[str]
) -> list[tuple[str, str, Claim]]:
    """Return ``(tool, reason, claim)`` for each claim with no successful matching call.

    ``reason`` is "tool_failed" when a candidate tool was called but errored, else
    "not_called". Not judged: claims that match no known tool, and, for "not_called",
    claims matched on the verb alone ("I've raised this with our team" against
    ``create_ticket``) or whose action another tool that ran may have performed
    ("I've removed the mug" after ``remove_item``, with ``cancel_order`` known).
    """
    fabricated = []
    for claim in claims:
        if not claim.candidates or any(c in ok_tools for c in claim.candidates):
            continue
        if any(c in error_tools for c in claim.candidates):
            fabricated.append((claim.candidates[0], "tool_failed", claim))
        elif claim.object_matched is not False and not family_ran(claim.verb, ok_tools):
            fabricated.append((claim.candidates[0], "not_called", claim))
    return fabricated


def _scorer(
    *,
    outputs: list[ChatCompletionMessage],
    tools: list[ToolSpec],
    failed_tool_call_ids: list[str],
    **kwargs: Any,
):
    ok_tools: set[str] = set()
    error_tools: set[str] = set()
    for message in outputs:
        if message["role"] != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            name = tool_call["function"]["name"]
            if tool_call.get("id") in failed_tool_call_ids:
                error_tools.add(name)
            else:
                ok_tools.add(name)
    known_tools = list(tools)
    known_names = {spec.name for spec in known_tools}
    known_tools += normalize_tool_specs(sorted((ok_tools | error_tools) - known_names))
    claims = extract_claims(_final_reply(outputs), known_tools)
    fabricated = _find_fabricated(claims, ok_tools, error_tools)
    if not fabricated:
        return True, None
    return False, ", ".join(
        f'{tool} ({reason}): "{claim.text}"' for tool, reason, claim in fabricated
    )


def create_action_hallucination_evaluator(
    *,
    tools: Optional[Sequence[Union[str, dict]]] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that checks whether the agent's final message claims actions it never performed.

    The final assistant message (the last message with the assistant role) is scanned
    with deterministic rules for completed-action claims ("I've emailed you", "Your
    refund has been issued") and each claim is matched to the tool that would have
    performed it. The score is False when any claim has no successful matching tool call
    in the trajectory, either because the tool was never called or because every call to
    it failed. No reference trajectory is needed. An empty trajectory, or one with no
    assistant message, has no claims and scores True.

    Args:
        tools (Optional[Sequence[Union[str, dict]]]): The tools the agent had available, as
            names or dicts in OpenAI function format or with "name" and "description" keys.
            Without it, only tools that appear in the trajectory's tool calls are known, so
            a claim about a tool the agent never called cannot be judged. Raises
            ValueError for a tool that is neither a name nor a dict with a name.

    Returns:
        SimpleEvaluator: A function that evaluates a trajectory for fabricated action claims

    The returned evaluator accepts:
        - outputs: List of messages or dict representing the model output trajectory
        - **kwargs: Additional arguments passed to the underlying evaluator

    A tool call counts as failed when its LangChain `ToolMessage` has `status="error"`,
    or when an OpenAI-format tool message has `"status": "error"`. OpenAI-format tool
    messages normally carry no status, so with those inputs every tool call that was made
    is treated as successful. A tool call with no tool response also counts as successful,
    and an OpenAI-format tool message with `"status": "error"` but no `tool_call_id` is
    ignored, since it cannot be paired with its call.

    Claims matched to a tool only by their verb ("I've raised this with our team" against
    `create_ticket`) are not flagged when that tool was never called, nor are claims whose
    action another tool that ran may have performed. They are flagged when the tool was
    called and failed.

    The result's comment lists each fabricated claim as `tool_name (reason): "claim text"`,
    where reason is `not_called` or `tool_failed`.

    Example:
    ```python
    evaluator = create_action_hallucination_evaluator(
        tools=["send_email", "lookup_order", "issue_refund"],
    )
    result = evaluator(outputs=...)
    ```
    """
    known_tools = normalize_tool_specs(tools or [])

    def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ):
        failed_tool_call_ids = _failed_tool_call_ids(outputs)
        outputs = _normalize_to_openai_messages_list(outputs)
        return _run_evaluator(
            run_name="action_hallucination",
            scorer=_scorer,
            feedback_key="action_hallucination",
            outputs=outputs,
            tools=known_tools,
            failed_tool_call_ids=failed_tool_call_ids,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_action_hallucination_evaluator(
    *,
    tools: Optional[Sequence[Union[str, dict]]] = None,
) -> SimpleAsyncEvaluator:
    """Creates an async evaluator that checks whether the agent's final message claims actions it never performed.

    The final assistant message (the last message with the assistant role) is scanned
    with deterministic rules for completed-action claims ("I've emailed you", "Your
    refund has been issued") and each claim is matched to the tool that would have
    performed it. The score is False when any claim has no successful matching tool call
    in the trajectory, either because the tool was never called or because every call to
    it failed. No reference trajectory is needed. An empty trajectory, or one with no
    assistant message, has no claims and scores True.

    Args:
        tools (Optional[Sequence[Union[str, dict]]]): The tools the agent had available, as
            names or dicts in OpenAI function format or with "name" and "description" keys.
            Without it, only tools that appear in the trajectory's tool calls are known, so
            a claim about a tool the agent never called cannot be judged. Raises
            ValueError for a tool that is neither a name nor a dict with a name.

    Returns:
        SimpleAsyncEvaluator: An async function that evaluates a trajectory for fabricated action claims

    The returned evaluator accepts:
        - outputs: List of messages or dict representing the model output trajectory
        - **kwargs: Additional arguments passed to the underlying evaluator

    A tool call counts as failed when its LangChain `ToolMessage` has `status="error"`,
    or when an OpenAI-format tool message has `"status": "error"`. OpenAI-format tool
    messages normally carry no status, so with those inputs every tool call that was made
    is treated as successful. A tool call with no tool response also counts as successful,
    and an OpenAI-format tool message with `"status": "error"` but no `tool_call_id` is
    ignored, since it cannot be paired with its call.

    Claims matched to a tool only by their verb ("I've raised this with our team" against
    `create_ticket`) are not flagged when that tool was never called, nor are claims whose
    action another tool that ran may have performed. They are flagged when the tool was
    called and failed.

    The result's comment lists each fabricated claim as `tool_name (reason): "claim text"`,
    where reason is `not_called` or `tool_failed`.

    Example:
    ```python
    evaluator = create_async_action_hallucination_evaluator(
        tools=["send_email", "lookup_order", "issue_refund"],
    )
    result = await evaluator(outputs=...)
    ```
    """
    known_tools = normalize_tool_specs(tools or [])

    async def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ):
        failed_tool_call_ids = _failed_tool_call_ids(outputs)
        outputs = _normalize_to_openai_messages_list(outputs)
        return await _arun_evaluator(
            run_name="action_hallucination",
            scorer=_scorer,
            feedback_key="action_hallucination",
            outputs=outputs,
            tools=known_tools,
            failed_tool_call_ids=failed_tool_call_ids,
            **kwargs,
        )

    return _wrapped_evaluator
