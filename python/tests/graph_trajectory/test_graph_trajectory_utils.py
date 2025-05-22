import pytest

from agentevals.graph_trajectory.utils import (
    aextract_langgraph_trajectory_from_thread,
    extract_langgraph_trajectory_from_thread,
)
from openevals.exact import exact_match

from typing import Annotated
from typing_extensions import TypedDict
import operator
import time

from langgraph.types import Command, Send
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


@pytest.mark.langsmith
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
@pytest.mark.langsmith
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


@pytest.mark.langsmith
def test_trajectory_match_with_command():
    checkpointer = MemorySaver()

    class State(TypedDict):
        items: Annotated[list[str], lambda x, y: x + y]
        processedCount: Annotated[int, lambda x, y: y]

    def dispatcher(state: State):
        # Use Command with Send to route to multiple processing nodes dynamically
        sends = [
            Send(f"process_{index % 2}", {"items": [item], "index": index})
            for index, item in enumerate(state["items"])
        ]
        return Command(
            update={"processedCount": len(state["items"])},
            goto=sends,
        )

    def process_0(state: State):
        return {"items": [f"processed_0: {', '.join(state['items'])}"]}

    def process_1(state: State):
        return {"items": [f"processed_1: {', '.join(state['items'])}"]}

    def aggregator(state: State):
        return {"items": [f"final count: {state['processedCount']}"]}

    graph = StateGraph(State)
    graph.add_node("dispatcher", dispatcher)
    graph.add_node("process_0", process_0)
    graph.add_node("process_1", process_1)
    graph.add_node("aggregator", aggregator)

    graph.add_edge("__start__", "dispatcher")
    graph.add_edge(["process_0", "process_1"], "aggregator")

    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "3"}}

    app.invoke(
        {
            "items": ["task1", "task2", "task3"],
        },
        config,
    )

    trajectory = extract_langgraph_trajectory_from_thread(app, config)

    assert exact_match(
        outputs=trajectory,
        reference_outputs={
            "inputs": [
                {
                    "__start__": {
                        "items": ["task1", "task2", "task3"],
                    },
                },
            ],
            "outputs": {
                "inputs": [],
                "results": [
                    {
                        "items": [
                            "task1",
                            "task2",
                            "task3",
                            "processed_0: task1",
                            "processed_1: task2",
                            "processed_0: task3",
                            "final count: 3",
                        ],
                        "processedCount": 3,
                    },
                ],
                "steps": [
                    [
                        "__start__",
                        "dispatcher",
                        "process_0",
                        "process_1",
                        "process_0",
                        "aggregator",
                    ],
                ],
            },
        },
    )["score"]
