from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import warnings

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages

from agentevals.types import GraphTrajectory, ExtractedLangGraphThreadTrajectory

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from langgraph.pregel import Pregel
    from langgraph.pregel.types import StateSnapshot


def extract_langgraph_trajectory_from_snapshots(
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
            if has_interrupts:
                trajectory["results"].append({})
            elif (
                isinstance(snapshot.values, dict)
                and "messages" in snapshot.values
                and isinstance(snapshot.values["messages"], list)
            ):
                # Just append the last message in the output to the results to reduce context size
                last_message = snapshot.values["messages"][-1]
                if isinstance(last_message, BaseMessage):
                    trajectory["results"].append(
                        {"messages": convert_to_openai_messages([last_message])}
                    )
                else:
                    trajectory["results"].append({"messages": [last_message]})
            else:
                trajectory["results"].append(snapshot.values)
            trajectory["steps"].append([])
        if is_acc_steps and snapshot.tasks:
            checkpoint_ns = snapshot.config.get("configurable", {}).get(
                "checkpoint_ns", ""
            )
            subgraph_path = ""
            if checkpoint_ns and len(checkpoint_ns.split(":")) > 1:
                subgraph_path = f"{checkpoint_ns.split(':')[0]}:"
            for task in snapshot.tasks:
                if task.interrupts:
                    trajectory["steps"][-1].append("__interrupt__")
                trajectory["steps"][-1].append(f"{subgraph_path}{task.name}")
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


def _get_langgraph_state_history_recursive(graph: Pregel, config: RunnableConfig):
    state_history = []
    for history in graph.get_state_history(config=config):
        if history.tasks:
            for task in history.tasks:
                if task.state and task.state.get("configurable", {}).get(
                    "checkpoint_ns", None
                ):
                    state_history.extend(
                        _get_langgraph_state_history_recursive(graph, task.state)
                    )
        state_history.append(history)
    return state_history


async def _aget_langgraph_state_history_recursive(
    graph: Pregel, config: RunnableConfig
):
    state_history = []
    async for history in graph.aget_state_history(config=config):
        if history.tasks:
            for task in history.tasks:
                if task.state and task.state.get("configurable", {}).get(
                    "checkpoint_ns", None
                ):
                    state_history.extend(
                        await _aget_langgraph_state_history_recursive(graph, task.state)
                    )
        state_history.append(history)
    return state_history


def extract_langgraph_trajectory_from_thread(
    graph: Pregel, config: RunnableConfig
) -> ExtractedLangGraphThreadTrajectory:
    return extract_langgraph_trajectory_from_snapshots(
        _get_langgraph_state_history_recursive(graph, config)
    )


async def aextract_langgraph_trajectory_from_thread(
    graph: Pregel, config: RunnableConfig
) -> ExtractedLangGraphThreadTrajectory:
    return extract_langgraph_trajectory_from_snapshots(
        await _aget_langgraph_state_history_recursive(graph, config)
    )
