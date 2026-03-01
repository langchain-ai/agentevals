from typing import Any, Literal

from agentevals.graph_trajectory.strict import _scorer as graph_trajectory_strict_scorer
from agentevals.graph_trajectory.unordered import (
    _scorer as graph_trajectory_unordered_scorer,
)
from agentevals.graph_trajectory.subset import (
    _scorer as graph_trajectory_subset_scorer,
)
from agentevals.graph_trajectory.superset import (
    _scorer as graph_trajectory_superset_scorer,
)
from agentevals.types import (
    EvaluatorResult,
    GraphTrajectory,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
)
from agentevals.utils import _run_evaluator, _arun_evaluator


GraphTrajectoryMatchMode = Literal["strict", "unordered", "subset", "superset"]


def create_graph_trajectory_match_evaluator(
    *,
    trajectory_match_mode: GraphTrajectoryMatchMode = "strict",
) -> SimpleEvaluator:
    """Creates an evaluator that compares graph trajectories between outputs and references.

    Unlike agent trajectory match evaluators which compare tool calls within messages,
    graph trajectory match evaluators compare the nodes visited at each step.

    Args:
        trajectory_match_mode (GraphTrajectoryMatchMode): The mode for matching trajectories:
            - "strict": Requires exact match in order and content for each step
            - "unordered": Allows matching nodes in any order across all steps
            - "subset": Accepts if output nodes are a subset of reference nodes
            - "superset": Accepts if output nodes are a superset of reference nodes

    Returns:
        SimpleEvaluator: A function that evaluates graph trajectory matches

    The returned evaluator accepts:
        - outputs: GraphTrajectory representing the actual agent execution
        - reference_outputs: GraphTrajectory representing the expected execution
        - **kwargs: Additional arguments passed to the underlying evaluator

    Example:
    ```python
    from agentevals.graph_trajectory.match import create_graph_trajectory_match_evaluator

    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode="unordered",
    )
    result = evaluator(
        outputs={
            "results": [],
            "steps": [["__start__", "agent", "tools"], ["agent"]],
        },
        reference_outputs={
            "results": [],
            "steps": [["__start__", "tools", "agent"], ["agent"]],
        },
    )
    ```
    """
    if trajectory_match_mode == "strict":
        scorer = graph_trajectory_strict_scorer
    elif trajectory_match_mode == "unordered":
        scorer = graph_trajectory_unordered_scorer
    elif trajectory_match_mode == "subset":
        scorer = graph_trajectory_subset_scorer
    elif trajectory_match_mode == "superset":
        scorer = graph_trajectory_superset_scorer
    else:
        raise ValueError(
            f"Invalid trajectory match mode: `{trajectory_match_mode}`. "
            "Must be one of `strict`, `unordered`, `subset`, or `superset`."
        )

    def _wrapped_evaluator(
        *,
        outputs: GraphTrajectory,
        reference_outputs: GraphTrajectory,
        **kwargs: Any,
    ) -> EvaluatorResult:
        return _run_evaluator(
            run_name=f"graph_trajectory_{trajectory_match_mode}_match",
            scorer=scorer,
            feedback_key=f"graph_trajectory_{trajectory_match_mode}_match",
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_graph_trajectory_match_evaluator(
    *,
    trajectory_match_mode: GraphTrajectoryMatchMode = "strict",
) -> SimpleAsyncEvaluator:
    """Creates an async evaluator that compares graph trajectories between outputs and references.

    Unlike agent trajectory match evaluators which compare tool calls within messages,
    graph trajectory match evaluators compare the nodes visited at each step.

    Args:
        trajectory_match_mode (GraphTrajectoryMatchMode): The mode for matching trajectories:
            - "strict": Requires exact match in order and content for each step
            - "unordered": Allows matching nodes in any order across all steps
            - "subset": Accepts if output nodes are a subset of reference nodes
            - "superset": Accepts if output nodes are a superset of reference nodes

    Returns:
        SimpleAsyncEvaluator: An async function that evaluates graph trajectory matches

    The returned evaluator accepts:
        - outputs: GraphTrajectory representing the actual agent execution
        - reference_outputs: GraphTrajectory representing the expected execution
        - **kwargs: Additional arguments passed to the underlying evaluator

    Example:
    ```python
    from agentevals.graph_trajectory.match import create_async_graph_trajectory_match_evaluator

    evaluator = create_async_graph_trajectory_match_evaluator(
        trajectory_match_mode="unordered",
    )
    result = await evaluator(
        outputs={
            "results": [],
            "steps": [["__start__", "agent", "tools"], ["agent"]],
        },
        reference_outputs={
            "results": [],
            "steps": [["__start__", "tools", "agent"], ["agent"]],
        },
    )
    ```
    """
    if trajectory_match_mode == "strict":
        scorer = graph_trajectory_strict_scorer
    elif trajectory_match_mode == "unordered":
        scorer = graph_trajectory_unordered_scorer
    elif trajectory_match_mode == "subset":
        scorer = graph_trajectory_subset_scorer
    elif trajectory_match_mode == "superset":
        scorer = graph_trajectory_superset_scorer
    else:
        raise ValueError(
            f"Invalid trajectory match mode: `{trajectory_match_mode}`. "
            "Must be one of `strict`, `unordered`, `subset`, or `superset`."
        )

    async def _wrapped_evaluator(
        *,
        outputs: GraphTrajectory,
        reference_outputs: GraphTrajectory,
        **kwargs: Any,
    ) -> EvaluatorResult:
        return await _arun_evaluator(
            run_name=f"graph_trajectory_{trajectory_match_mode}_match",
            scorer=scorer,
            feedback_key=f"graph_trajectory_{trajectory_match_mode}_match",
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator
