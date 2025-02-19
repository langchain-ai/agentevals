from agentevals.graph_trajectory.utils import (
    extract_langgraph_trajectory_from_thread,
)
from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

import pytest


@tool
def search(query: str):
    """Call to surf the web."""
    user_answer = interrupt("Tell me the answer to the question.")
    return user_answer


tools = [search]


@pytest.mark.langsmith
def test_trajectory_match():
    checkpointer = MemorySaver()
    graph = create_react_agent(
        model="gpt-4o-mini",
        checkpointer=checkpointer,
        tools=[search],
    )
    graph.invoke(
        {"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
        config={"configurable": {"thread_id": "1"}},
    )
    graph.invoke(
        Command(resume="It is rainy and 70 degrees!"),
        config={"configurable": {"thread_id": "1"}},
    )
    extracted_trajectory = extract_langgraph_trajectory_from_thread(
        graph, {"configurable": {"thread_id": "1"}}
    )
    reference_trajectory = {
        "results": [],
        "steps": [["__start__", "agent", "tools", "__interrupt__"], ["agent"]],
    }
    res = graph_trajectory_strict_match(
        outputs=extracted_trajectory["outputs"],
        reference_outputs=reference_trajectory,
    )
    assert res["score"]
