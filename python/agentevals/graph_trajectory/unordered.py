from __future__ import annotations

from agentevals.types import GraphTrajectory
from agentevals.graph_trajectory.utils import _is_graph_trajectory_superset

from typing import Any


def _scorer(
    *,
    outputs: GraphTrajectory,
    reference_outputs: GraphTrajectory,
    **kwargs: Any,
) -> bool:
    if outputs is None or reference_outputs is None:
        raise ValueError(
            "Graph trajectory unordered match requires both outputs and reference_outputs"
        )
    return _is_graph_trajectory_superset(
        outputs, reference_outputs
    ) and _is_graph_trajectory_superset(
        reference_outputs, outputs
    )
