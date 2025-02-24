from agentevals.graph_trajectory.utils import (
    aextract_langgraph_trajectory_from_thread,
)
from agentevals.graph_trajectory.llm import create_async_graph_trajectory_llm_as_judge

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langchain import hub

import pytest


@tool
def search(query: str):
    """Call to surf the web."""
    user_answer = interrupt("Tell me the answer to the question.")
    return user_answer


tools = [search]

@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_use_prompt_hub_prompt():
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
    extracted_trajectory = await aextract_langgraph_trajectory_from_thread(
        graph, {"configurable": {"thread_id": "1"}}
    )
    evaluator = create_async_graph_trajectory_llm_as_judge(
        model="openai:o3-mini",
        prompt=hub.pull("langchain-ai/test-trajectory")
    )
    res = await evaluator(
        inputs=extracted_trajectory["inputs"],
        outputs=extracted_trajectory["outputs"],
    )
    assert res["key"] == "graph_trajectory_accuracy"
    assert res["score"]

@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_sensible_trajectory():
    checkpointer = MemorySaver()
    graph = create_react_agent(
        model="gpt-4o-mini",
        checkpointer=checkpointer,
        tools=[search],
    )
    await graph.ainvoke(
        {"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
        config={"configurable": {"thread_id": "1"}},
    )
    await graph.ainvoke(
        Command(resume="It is rainy and 70 degrees!"),
        config={"configurable": {"thread_id": "1"}},
    )
    extracted_trajectory = await aextract_langgraph_trajectory_from_thread(
        graph, {"configurable": {"thread_id": "1"}}
    )
    evaluator = create_async_graph_trajectory_llm_as_judge(
        model="openai:o3-mini",
    )
    res = await evaluator(
        inputs=extracted_trajectory["inputs"],
        outputs=extracted_trajectory["outputs"],
    )
    assert res["key"] == "graph_trajectory_accuracy"
    assert res["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_unsensible_trajectory():
    checkpointer = MemorySaver()

    @tool
    def askjeeves(query: str):
        """Call to surf the web."""
        return "foo"

    graph = create_react_agent(
        model="gpt-4o-mini",
        checkpointer=checkpointer,
        tools=[askjeeves],
        prompt="You are an evil assistant who is inefficient and calls more tools than necessary.",
    )
    await graph.ainvoke(
        {"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
        config={"configurable": {"thread_id": "1"}},
    )
    extracted_trajectory = await aextract_langgraph_trajectory_from_thread(
        graph, {"configurable": {"thread_id": "1"}}
    )
    evaluator = create_async_graph_trajectory_llm_as_judge(
        prompt="""You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal steps in resolving a user queries.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is perfectly efficient, with no more than one tool call
  - Is semantically equivalent to the provided reference trajectory, if present
</Rubric>

<Instructions>
  Grade the following thread, evaluating whether the agent's overall steps are logical and relatively efficient.
  For the trajectory, "__start__" denotes an initial entrypoint to the agent, and "__interrupt__" corresponds to the agent
  interrupting to await additional data from another source ("human-in-the-loop"):
</Instructions>

<thread>
{thread}
</thread>

{reference_outputs}
""",
        model="openai:o3-mini",
    )
    res = await evaluator(
        inputs=extracted_trajectory["inputs"],
        outputs=extracted_trajectory["outputs"],
    )
    assert res["key"] == "graph_trajectory_accuracy"
    assert not res["score"]
