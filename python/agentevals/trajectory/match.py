from typing import Literal, Optional, Union

from agentevals.trajectory.strict import _scorer as trajectory_strict_scorer
from agentevals.trajectory.unordered import _scorer as trajectory_unordered_scorer
from agentevals.trajectory.subset import _scorer as trajectory_subset_scorer
from agentevals.trajectory.superset import _scorer as trajectory_superset_scorer
from agentevals.types import (
    ChatCompletionMessage,
    EvaluatorResult,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    ToolArgsMatchMode,
    ToolArgsMatchOverrides,
)
from agentevals.utils import _run_evaluator, _arun_evaluator

from openevals.utils import (
    _normalize_to_openai_messages_list,
)

from langchain_core.messages import BaseMessage


def create_trajectory_match_evaluator(
    *,
    trajectory_match_mode: Literal[
        "strict", "unordered", "subset", "superset"
    ] = "strict",
    tool_args_match_mode: ToolArgsMatchMode = "exact",
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
    **kwargs,
) -> SimpleEvaluator:
    if trajectory_match_mode == "strict":
        scorer = trajectory_strict_scorer
    elif trajectory_match_mode == "unordered":
        scorer = trajectory_unordered_scorer
    elif trajectory_match_mode == "subset":
        scorer = trajectory_subset_scorer
    elif trajectory_match_mode == "superset":
        scorer = trajectory_superset_scorer
    else:
        raise ValueError(f"Invalid trajectory match type: {trajectory_match_mode}")

    def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ) -> EvaluatorResult:
        outputs = _normalize_to_openai_messages_list(outputs)
        reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
        return _run_evaluator(
            run_name=f"trajectory_{trajectory_match_mode}_match",
            scorer=scorer,
            feedback_key=f"trajectory_{trajectory_match_mode}_match",
            outputs=outputs,
            reference_outputs=reference_outputs,
            tool_args_match_mode=tool_args_match_mode,
            tool_args_match_overrides=tool_args_match_overrides,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_trajectory_match_evaluator(
    *,
    trajectory_match_mode: Literal[
        "strict", "unordered", "subset", "superset"
    ] = "strict",
    tool_args_match_mode: ToolArgsMatchMode = "exact",
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
    **kwargs,
) -> SimpleAsyncEvaluator:
    if trajectory_match_mode == "strict":
        scorer = trajectory_strict_scorer
    elif trajectory_match_mode == "unordered":
        scorer = trajectory_unordered_scorer
    elif trajectory_match_mode == "subset":
        scorer = trajectory_subset_scorer
    elif trajectory_match_mode == "superset":
        scorer = trajectory_superset_scorer
    else:
        raise ValueError(f"Invalid trajectory match type: {trajectory_match_mode}")

    async def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ) -> EvaluatorResult:
        outputs = _normalize_to_openai_messages_list(outputs)
        reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
        return await _arun_evaluator(
            run_name=f"trajectory_{trajectory_match_mode}_match",
            scorer=scorer,
            feedback_key=f"trajectory_{trajectory_match_mode}_match",
            outputs=outputs,
            reference_outputs=reference_outputs,
            tool_args_match_mode=tool_args_match_mode,
            tool_args_match_overrides=tool_args_match_overrides,
            **kwargs,
        )

    return _wrapped_evaluator
