from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Iterable, List, TypedDict, TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from langgraph.pregel import Pregel
    from langgraph.pregel.types import StateSnapshot


class TestCase(TypedDict):
    id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any]


def _langgraph_trajectory(snapshots: Iterable[StateSnapshot]) -> TestCase:
    test_case = TestCase(
        id=None,
        inputs={
            "input": [],
        },
        outputs={
            "output": [],
            "steps": [],
        },
    )
    is_acc_steps = False
    for snapshot in snapshots:
        if not test_case["id"]:
            test_case["id"] = snapshot.config["configurable"]["thread_id"]
        if not snapshot.next:
            is_acc_steps = True
            test_case["outputs"]["output"].append(snapshot.values)
            test_case["outputs"]["steps"].append([])
            if not test_case.get("metadata"):
                test_case["metadata"] = snapshot.config["configurable"]
        if (
            is_acc_steps
            and snapshot.metadata["source"] == "loop"
            and snapshot.metadata["writes"]
        ):
            for node in snapshot.metadata["writes"]:
                test_case["outputs"]["steps"][-1].append(node)
        if is_acc_steps and snapshot.metadata["source"] == "input":
            test_case["inputs"]["input"].append(snapshot.metadata["writes"])
    test_case["inputs"]["input"].reverse()
    test_case["outputs"]["output"].reverse()
    test_case["outputs"]["steps"].reverse()
    for ss in test_case["outputs"]["steps"]:
        ss.reverse()
    return test_case


def extract_langgraph_trajectory_from_thread(
    graph: Pregel, config: RunnableConfig
) -> TestCase:
    return _langgraph_trajectory(graph.get_state_history(config))


async def aextract_langgraph_trajectory_from_thread(
    graph: Pregel, config: RunnableConfig
) -> TestCase:
    return _langgraph_trajectory([s async for s in graph.get_state_history(config)])
