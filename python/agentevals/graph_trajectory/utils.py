from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import warnings

from langchain_core.messages import BaseMessage

from agentevals.types import GraphTrajectory, ExtractedLangGraphThreadTrajectory

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from langgraph.pregel import Pregel
    from langgraph.pregel.types import StateSnapshot


def _extract_langgraph_trajectory_from_snapshots(
    snapshots: Iterable[StateSnapshot],
) -> ExtractedLangGraphThreadTrajectory:
    inputs = []
    trajectory = GraphTrajectory(
        inputs=[],
        results=[],
        steps=[],
    )
    is_acc_steps = False
    snapshot_list = list(snapshots)
    for i, snapshot in enumerate(snapshot_list):
        has_interrupts = any(t.interrupts for t in snapshot.tasks)
        if not snapshot.next or has_interrupts:
            is_acc_steps = True
            if (
                isinstance(snapshot.values, dict)
                and "messages" in snapshot.values
                and isinstance(snapshot.values["messages"], list)
            ):
                if has_interrupts:
                    trajectory["results"].append({})
                else:
                    # Just append the last message in the output to the results to reduce context size
                    last_message = snapshot.values["messages"][-1]
                    if isinstance(last_message, BaseMessage):
                        trajectory["results"].append(
                            {
                                "messages": [
                                    {
                                        "role": last_message.type,
                                        "content": last_message.content,
                                    }
                                ]
                            }
                        )
                    else:
                        trajectory["results"].append({"messages": [last_message]})
            else:
                trajectory["results"].append(snapshot.values)
            trajectory["steps"].append([])
        if is_acc_steps and snapshot.tasks:
            for task in snapshot.tasks:
                if task.interrupts:
                    trajectory["steps"][-1].append("__interrupt__")
                trajectory["steps"][-1].append(task.name)
        if is_acc_steps:
            if snapshot.metadata is not None and snapshot.metadata["source"] == "input":
                inputs.append(snapshot.metadata["writes"])
            elif i + 1 < len(snapshot_list) and any(
                t.interrupts for t in snapshot_list[i + 1].tasks
            ):
                inputs.append("__resuming__")  # type: ignore
    inputs.reverse()
    trajectory["results"].reverse()
    trajectory["steps"].reverse()
    for ss in trajectory["steps"]:
        ss.reverse()
    if len(inputs) != len(trajectory["results"]):
        warnings.warn(
            "Trajectory parsing may be incomplete: inputs and results have different lengths"
        )
    elif len(inputs) != len(trajectory["steps"]):
        warnings.warn(
            "Trajectory parsing may be incomplete: inputs and steps have different lengths"
        )

    return {"inputs": inputs, "outputs": trajectory}


def extract_langgraph_trajectory_from_thread(
    graph: Pregel, config: RunnableConfig
) -> ExtractedLangGraphThreadTrajectory:
    return _extract_langgraph_trajectory_from_snapshots(graph.get_state_history(config))


async def aextract_langgraph_trajectory_from_thread(
    graph: Pregel, config: RunnableConfig
) -> ExtractedLangGraphThreadTrajectory:
    return _extract_langgraph_trajectory_from_snapshots(
        [s async for s in graph.aget_state_history(config)]
    )
