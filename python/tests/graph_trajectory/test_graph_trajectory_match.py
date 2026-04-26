from agentevals.graph_trajectory.match import (
    create_graph_trajectory_match_evaluator,
    create_async_graph_trajectory_match_evaluator,
)
from agentevals.types import EvaluatorResult

import pytest


# ---------------------------------------------------------------------------
# Exact same trajectory → all modes should pass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("graph_trajectory_strict_match", "strict"),
        ("graph_trajectory_unordered_match", "unordered"),
        ("graph_trajectory_subset_match", "subset"),
        ("graph_trajectory_superset_match", "superset"),
    ],
)
def test_identical_trajectories(feedback_key, match_mode):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "__interrupt__"], ["agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "__interrupt__"], ["agent"]],
    }
    assert evaluator(outputs=outputs, reference_outputs=reference_outputs) == (
        EvaluatorResult(key=feedback_key, score=True, comment=None, metadata=None)
    )


# ---------------------------------------------------------------------------
# Same nodes, different order across turns → strict fails, others pass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("graph_trajectory_strict_match", "strict", 0.0),
        ("graph_trajectory_unordered_match", "unordered", 1.0),
        ("graph_trajectory_subset_match", "subset", 1.0),
        ("graph_trajectory_superset_match", "superset", 1.0),
    ],
)
def test_same_nodes_different_order(feedback_key, match_mode, score):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "tools", "agent"], ["agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    assert evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


# ---------------------------------------------------------------------------
# Same nodes split across different number of turns → strict fails, unordered passes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("graph_trajectory_strict_match", "strict", 0.0),
        ("graph_trajectory_unordered_match", "unordered", 1.0),
        ("graph_trajectory_subset_match", "subset", 1.0),
        ("graph_trajectory_superset_match", "superset", 1.0),
    ],
)
def test_same_nodes_different_turns(feedback_key, match_mode, score):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    # Two turns
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    # One turn with same nodes
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "agent"]],
    }
    assert evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


# ---------------------------------------------------------------------------
# Output has extra nodes → superset passes, subset / unordered / strict fail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("graph_trajectory_strict_match", "strict", 0.0),
        ("graph_trajectory_unordered_match", "unordered", 0.0),
        ("graph_trajectory_subset_match", "subset", 0.0),
        ("graph_trajectory_superset_match", "superset", 1.0),
    ],
)
def test_output_has_extra_nodes(feedback_key, match_mode, score):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "retriever"], ["agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    assert evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


# ---------------------------------------------------------------------------
# Output has fewer nodes → subset passes, superset / unordered / strict fail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("graph_trajectory_strict_match", "strict", 0.0),
        ("graph_trajectory_unordered_match", "unordered", 0.0),
        ("graph_trajectory_subset_match", "subset", 1.0),
        ("graph_trajectory_superset_match", "superset", 0.0),
    ],
)
def test_output_has_fewer_nodes(feedback_key, match_mode, score):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    assert evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


# ---------------------------------------------------------------------------
# Completely different nodes → all modes fail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("graph_trajectory_strict_match", "strict"),
        ("graph_trajectory_unordered_match", "unordered"),
        ("graph_trajectory_subset_match", "subset"),
        ("graph_trajectory_superset_match", "superset"),
    ],
)
def test_completely_different_nodes(feedback_key, match_mode):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "planner", "executor"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"]],
    }
    assert evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=False, comment=None, metadata=None)


# ---------------------------------------------------------------------------
# Both empty → all modes pass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("graph_trajectory_strict_match", "strict"),
        ("graph_trajectory_unordered_match", "unordered"),
        ("graph_trajectory_subset_match", "subset"),
        ("graph_trajectory_superset_match", "superset"),
    ],
)
def test_both_empty(feedback_key, match_mode):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {"results": [], "steps": []}
    reference_outputs = {"results": [], "steps": []}
    assert evaluator(outputs=outputs, reference_outputs=reference_outputs) == (
        EvaluatorResult(key=feedback_key, score=True, comment=None, metadata=None)
    )


# ---------------------------------------------------------------------------
# Subgraph nodes (e.g. "inner:inner_1") are treated as distinct nodes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("graph_trajectory_strict_match", "strict"),
        ("graph_trajectory_unordered_match", "unordered"),
        ("graph_trajectory_subset_match", "subset"),
        ("graph_trajectory_superset_match", "superset"),
    ],
)
def test_subgraph_nodes(feedback_key, match_mode):
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [
            [
                "__start__",
                "outer_1",
                "inner",
                "inner:__start__",
                "inner:inner_1",
                "inner:inner_2",
            ],
            ["outer_2"],
        ],
    }
    reference_outputs = {
        "results": [],
        "steps": [
            [
                "__start__",
                "outer_1",
                "inner",
                "inner:__start__",
                "inner:inner_1",
                "inner:inner_2",
            ],
            ["outer_2"],
        ],
    }
    assert evaluator(outputs=outputs, reference_outputs=reference_outputs) == (
        EvaluatorResult(key=feedback_key, score=True, comment=None, metadata=None)
    )


# ---------------------------------------------------------------------------
# Duplicate node names are counted correctly
# ---------------------------------------------------------------------------
def test_duplicate_nodes_unordered():
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode="unordered",
    )
    # Output calls "agent" twice, reference calls "agent" three times
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "agent", "agent"]],
    }
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert not result["score"]


def test_duplicate_nodes_superset():
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode="superset",
    )
    # Output has agent ×3, reference has agent ×2 → superset passes
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "agent", "agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "agent"]],
    }
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["score"]


# ---------------------------------------------------------------------------
# Invalid match mode raises ValueError
# ---------------------------------------------------------------------------
def test_invalid_match_mode():
    with pytest.raises(ValueError, match="Invalid trajectory match mode"):
        create_graph_trajectory_match_evaluator(trajectory_match_mode="fuzzy")


# ---------------------------------------------------------------------------
# Interrupt nodes are preserved in comparison
# ---------------------------------------------------------------------------
def test_interrupt_handling():
    evaluator = create_graph_trajectory_match_evaluator(
        trajectory_match_mode="strict",
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "__interrupt__"], ["agent"]],
    }
    # Missing __interrupt__
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert not result["score"]


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("graph_trajectory_strict_match", "strict"),
        ("graph_trajectory_unordered_match", "unordered"),
        ("graph_trajectory_subset_match", "subset"),
        ("graph_trajectory_superset_match", "superset"),
    ],
)
async def test_async_identical_trajectories(feedback_key, match_mode):
    evaluator = create_async_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    assert await evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=True, comment=None, metadata=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("graph_trajectory_strict_match", "strict", 0.0),
        ("graph_trajectory_unordered_match", "unordered", 0.0),
        ("graph_trajectory_subset_match", "subset", 0.0),
        ("graph_trajectory_superset_match", "superset", 1.0),
    ],
)
async def test_async_output_has_extra_nodes(feedback_key, match_mode, score):
    evaluator = create_async_graph_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
    )
    outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "retriever"], ["agent"]],
    }
    reference_outputs = {
        "results": [],
        "steps": [["__start__", "agent", "tools"], ["agent"]],
    }
    assert await evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)
