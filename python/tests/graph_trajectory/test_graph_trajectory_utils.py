import pytest

from agentevals.graph_trajectory.utils import (
    aextract_langgraph_trajectory_from_thread,
    extract_langgraph_trajectory_from_thread,
)
from openevals.exact import exact_match

from typing import Annotated, TypedDict
import operator
import time

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


def test_trajectory_match():
    checkpointer = MemorySaver()

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        time.sleep(0.1)
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        return {
            "my_key": " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def outer_1(state: State):
        return {"my_key": " and parallel"}

    def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile(interrupt_before=["inner_2"]))
    graph.add_node("outer_1", outer_1)
    graph.add_node("outer_2", outer_2)

    graph.add_edge("__start__", "inner")
    graph.add_edge("__start__", "outer_1")
    graph.add_edge(["inner", "outer_1"], "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({"my_key": ""}, config)

    app.invoke(None, config) == {
        "my_key": "got here and there and parallel and back again",
    }
    assert exact_match(
        outputs=extract_langgraph_trajectory_from_thread(
            app, {"configurable": {"thread_id": "1"}}
        ),
        reference_outputs={
            "inputs": [
                {"__start__": {"my_key": ""}},
                {"__start__": {"my_key": ""}},
            ],
            "outputs": {
                "inputs": [],
                "results": [
                    {"my_key": "got here and there", "my_other_key": "got here"},
                    {"my_key": "got here and there and parallel and back again"},
                ],
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
            },
        },
    )["score"]


@pytest.mark.asyncio
async def test_trajectory_match_async():
    checkpointer = MemorySaver()

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        time.sleep(0.1)
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        return {
            "my_key": " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def outer_1(state: State):
        return {"my_key": " and parallel"}

    def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile(interrupt_before=["inner_2"]))
    graph.add_node("outer_1", outer_1)
    graph.add_node("outer_2", outer_2)

    graph.add_edge("__start__", "inner")
    graph.add_edge("__start__", "outer_1")
    graph.add_edge(["inner", "outer_1"], "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({"my_key": ""}, config)

    await app.ainvoke(None, config) == {
        "my_key": "got here and there and parallel and back again",
    }
    assert exact_match(
        outputs=await aextract_langgraph_trajectory_from_thread(
            app, {"configurable": {"thread_id": "1"}}
        ),
        reference_outputs={
            "inputs": [
                {"__start__": {"my_key": ""}},
                {"__start__": {"my_key": ""}},
            ],
            "outputs": {
                "inputs": [],
                "results": [
                    {"my_key": "got here and there", "my_other_key": "got here"},
                    {"my_key": "got here and there and parallel and back again"},
                ],
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
            },
        },
    )["score"]
