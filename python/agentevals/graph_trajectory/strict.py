from __future__ import annotations

from agentevals.types import EvaluatorResult, GraphTrajectory
from agentevals.utils import _run_evaluator, _arun_evaluator

from typing import Any


def _scorer(
    *,
    outputs: GraphTrajectory,
    reference_outputs: GraphTrajectory,
) -> float:
    if outputs is None or reference_outputs is None:
        raise ValueError(
            "Strict trajectory match requires both outputs and reference_outputs"
        )
    if len(outputs["steps"]) != len(reference_outputs["steps"]):
        return False
    exact_match = True
    for output, reference_output in zip(outputs["steps"], reference_outputs["steps"]):
        if output != reference_output:
            exact_match = False
            break
    return exact_match


def graph_trajectory_strict_match(
    *,
    outputs: GraphTrajectory,
    reference_outputs: GraphTrajectory,
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an input graph trajectory strictly matches a reference graph trajectory.
    This means that at each step, the agent took the same steps in the same order as specified in the reference trajectory.

    Args:
        outputs (GraphTrajectory): Actual trajectory the agent followed.
        reference_outputs (GraphTrajectory): Ideal reference trajectory the agent should have followed.

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """
    return _run_evaluator(
        run_name="graph_trajectory_strict_match",
        scorer=_scorer,
        feedback_key="graph_trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
    )


async def graph_trajectory_strict_match_async(
    *,
    outputs: GraphTrajectory,
    reference_outputs: GraphTrajectory,
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an input graph trajectory strictly matches a reference graph trajectory.
    This means that at each step, the agent took the same steps in the same order as specified in the reference trajectory.

    Args:
        outputs (GraphTrajectory): Actual trajectory the agent followed.
        reference_outputs (GraphTrajectory): Ideal reference trajectory the agent should have followed.

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """

    async def async_wrapper(**kwargs: Any):
        return _scorer(**kwargs)

    return await _arun_evaluator(
        run_name="graph_trajectory_strict_match",
        scorer=async_wrapper,
        feedback_key="graph_trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
