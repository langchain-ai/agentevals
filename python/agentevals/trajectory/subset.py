from __future__ import annotations
from warnings import warn

from openevals.utils import (
    _normalize_to_openai_messages_list,
)
from agentevals.types import ChatCompletionMessage, EvaluatorResult
from agentevals.trajectory.utils import _is_trajectory_superset
from agentevals.utils import _run_evaluator, _arun_evaluator
from agentevals.types import ToolArgsMatchMode, ToolArgsMatchOverrides

from typing import Any, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _scorer(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    tool_args_match_mode: ToolArgsMatchMode,
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
    **kwargs: Any,
) -> EvaluatorResult:
    if outputs is None or reference_outputs is None:
        raise ValueError(
            "Trajectory subset match requires both outputs and reference_outputs"
        )
    is_superset = _is_trajectory_superset(
        reference_outputs, outputs, tool_args_match_mode, tool_args_match_overrides
    )
    return is_superset


def trajectory_subset(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    DEPRECATED: Use create_trajectory_match_evaluator() instead:
    ```python
    from agentevals.trajectory.match import create_trajectory_match_evaluator
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="subset")
    evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```

    Evaluate whether an agent trajectory and called tools is a subset of a reference trajectory and called tools.
    This means the agent called a subset of the tools specified in the reference trajectory.

    Args:
        outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Actual trajectory the agent followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        reference_outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Ideal reference trajectory the agent should have followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """
    warn(
        "trajectory_subset() is deprecated. Use create_trajectory_match_evaluator(trajectory_match_mode='subset') instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)

    return _run_evaluator(
        run_name="trajectory_subset",
        scorer=_scorer,
        feedback_key="trajectory_subset",
        outputs=outputs,
        reference_outputs=reference_outputs,
        tool_args_match_mode="none",
        **kwargs,
    )


async def trajectory_subset_async(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    DEPRECATED: Use create_async_trajectory_match_evaluator() instead:
    ```python
    from agentevals.trajectory.match import create_trajectory_match_evaluator
    evaluator = create_async_trajectory_match_evaluator(trajectory_match_mode="subset")
    await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```

    Evaluate whether an agent trajectory and called tools is a subset of a reference trajectory and called tools.
    This means the agent called a subset of the tools specified in the reference trajectory.

    Args:
        outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Actual trajectory the agent followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        reference_outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Ideal reference trajectory the agent should have followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """
    warn(
        "trajectory_subset_async() is deprecated. Use create_async_trajectory_match_evaluator(trajectory_match_mode='subset') instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)

    return await _arun_evaluator(
        run_name="trajectory_subset",
        scorer=_scorer,
        feedback_key="trajectory_subset",
        outputs=outputs,
        reference_outputs=reference_outputs,
        tool_args_match_mode="none",
        **kwargs,
    )
