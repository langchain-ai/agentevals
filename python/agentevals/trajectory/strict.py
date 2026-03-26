from __future__ import annotations
from warnings import warn
import json

from agentevals.trajectory.utils import _normalize_to_openai_messages_list
from agentevals.types import (
    ChatCompletionMessage,
    ToolArgsMatchMode,
    ToolArgsMatchOverrides,
)
from agentevals.utils import _run_evaluator, _arun_evaluator
from agentevals.trajectory.utils import _get_matcher_for_tool_name

from typing import Any, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _scorer(
    *,
    outputs: list[ChatCompletionMessage],
    reference_outputs: list[ChatCompletionMessage],
    tool_args_match_mode: ToolArgsMatchMode,
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
    **kwargs: Any,
):
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
    if not isinstance(outputs, list) or not isinstance(reference_outputs, list):
        raise ValueError(
            "Strict trajectory match requires both outputs and reference_outputs to be lists"
        )
    if len(outputs) != len(reference_outputs):
        error_message = f'Trajectory length mismatch: expected {len(reference_outputs)} steps but got {len(outputs)} steps.'
        return False, error_message
    for i, (output, reference_output) in enumerate(zip(outputs, reference_outputs)):
        if output["role"] != reference_output["role"]:
            error_message = f'Role mismatch at step {i}: expected {reference_output["role"]} but got {output["role"]}'
            return False, error_message
        elif ("tool_calls" in output and output["tool_calls"] is not None) != (
            "tool_calls" in reference_output
            and reference_output["tool_calls"] is not None
        ):
            # One has tool calls while the other doesn't
            output_has_tool_calls = "tool_calls" in output and output["tool_calls"] is not None
            reference_has_tool_calls = "tool_calls" in reference_output and reference_output["tool_calls"] is not None
            error_message = f'Tool call mismatch at step {i}: expected {"tool calls" if reference_has_tool_calls else "no tool calls"} but got {"tool calls" if output_has_tool_calls else "no tool calls"}'
            return False, error_message
        elif "tool_calls" in output and output["tool_calls"] is not None:
            # Both have tool calls, compare them
            if not isinstance(reference_output["tool_calls"], list):
                raise ValueError(
                    f'Invalid tool call format in reference trajectory at step {i}: expected a list of tool calls but got {type(reference_output["tool_calls"])}'
                )
            elif not isinstance(output["tool_calls"], list):
                error_message = f'Invalid tool call format at step {i}: expected a list of tool calls but got {type(output["tool_calls"])}'
                return False, error_message
            elif len(output["tool_calls"]) != len(reference_output["tool_calls"]):
                error_message = f'Tool call count mismatch at step {outputs.index(output)}: expected {len(reference_output["tool_calls"])} but got {len(output["tool_calls"])}'
                return False, error_message

            # Create a copy of reference tool calls to track matches
            seen = [False] * len(reference_output["tool_calls"])
            for output_call in output["tool_calls"]:
                found_match = False
                for i, reference_call in enumerate(reference_output["tool_calls"]):
                    if not seen[i] and (
                        output_call["function"]["name"]
                        == reference_call["function"]["name"]
                    ):
                        matcher = _get_matcher_for_tool_name(
                            output_call["function"]["name"],
                            tool_args_match_mode,
                            tool_args_match_overrides,
                        )
                        if matcher(
                            json.loads(output_call["function"]["arguments"]),
                            json.loads(reference_call["function"]["arguments"]),
                        ):
                            found_match = True
                            seen[i] = True
                            break
                if not found_match:
                    error_message = f'Tool call mismatch at step {i}: no matching tool call found for {output_call["function"]["name"]} with arguments {output_call["function"]["arguments"]}'
                    return False, error_message
    return True, None


def trajectory_strict_match(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    tool_call_args_exact_match: bool = True,
    **kwargs: Any,
):
    """
    DEPRECATED: Use create_trajectory_match_evaluator() instead:
    ```python
    from agentevals.trajectory.match import create_trajectory_match_evaluator
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```

    Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
    This means that at each step, the agent called the same tools in the same order as specified in the reference trajectory.

    Args:
        outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Actual trajectory the agent followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        reference_outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Ideal reference trajectory the agent should have followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        tool_call_args_exact_match (bool): Whether to require exact matches for tool call arguments

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """
    warn(
        "trajectory_strict_match() is deprecated. Use create_trajectory_match_evaluator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def wrapper(**kwargs: Any):
        return _scorer(
            tool_args_match_mode="exact" if tool_call_args_exact_match else "ignore",
            **kwargs,
        )

    return _run_evaluator(
        run_name="trajectory_strict_match",
        scorer=wrapper,
        feedback_key="trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
        **kwargs,
    )


async def trajectory_strict_match_async(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    tool_call_args_exact_match: bool = True,
    **kwargs: Any,
):
    """
    DEPRECATED: Use create_async_trajectory_match_evaluator() instead:
    ```python
    from agentevals.trajectory.match import create_trajectory_match_evaluator
    evaluator = create_async_trajectory_match_evaluator(trajectory_match_mode="subset")
    await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```

    Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
    This means that at each step, the agent called the same tools in the same order as specified in the reference trajectory.

    Args:
        outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Actual trajectory the agent followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        reference_outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Ideal reference trajectory the agent should have followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        tool_call_args_exact_match (bool): Whether to require exact matches for tool call arguments

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """
    warn(
        "trajectory_strict_match_async() is deprecated. Use create_async_trajectory_match_evaluator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def wrapper(**kwargs: Any):
        return _scorer(
            tool_args_match_mode="exact" if tool_call_args_exact_match else "ignore",
            **kwargs,
        )

    return await _arun_evaluator(
        run_name="trajectory_strict_match",
        scorer=wrapper,
        feedback_key="trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
        tool_call_args_exact_match=tool_call_args_exact_match,
        **kwargs,
    )
