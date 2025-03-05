from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from agentevals.trajectory.llm import create_trajectory_llm_as_judge
from openevals.llm import SimpleEvaluator

from typing import Optional


def generate_default_trajectory_prompt(criteria: str) -> str:
    return f"""
You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal trajectory and its final result when responding to a given input.

<Rubric>
  {criteria}
</Rubric>

First, try to understand the goal of the trajectory by looking at the input
(if the input is not present try to infer it from the content of the first message),
as well as the output of the final message. Once you understand the goal, grade the trajectory
as it relates to achieving that goal.

Grade the following trajectory:

<trajectory>
{{outputs}}
</trajectory>
{{inputs}}
"""


def wrap_graph_with_reflection(
    *, graph: Pregel, evaluator: Optional[SimpleEvaluator] = None
) -> StateGraph:
    class ReflectionAgentState(graph.builder.schema):
        agentevals_evaluation_criteria: str

    def generate_evaluation_criteria(
        state: ReflectionAgentState,
    ) -> ReflectionAgentState:
        inputs = state["messages"][0].content
        llm = init_chat_model("openai:o3-mini")
        res = llm.invoke(
            [
                {
                    "role": "system",
                    "content": """
You are a expert data labeler that generates evaluation criteria that measures whether a task is completed correctly.
The criteria you choose:

- Should be specific, measurable, and achievable.
- Should be concise and easy to understand.
- May take intermediate steps into account, but should prioritize the correctness of the overall end result over specific intermediate steps.
""",
                },
                {
                    "role": "user",
                    "content": f"""Generate criteria that would measure whether the following task has been solved correctly.
Respond with only the criteria and nothing else.

<task>
{inputs}
</task>
""",
                },
            ]
        )
        return {"agentevals_evaluation_criteria": res.content}

    def reflect(state: ReflectionAgentState) -> ReflectionAgentState:
        inputs = state["messages"][0].content
        outputs = state["messages"]
        nonlocal evaluator
        if evaluator is None:
            evaluator = create_trajectory_llm_as_judge(
                model="openai:o3-mini",
                prompt=generate_default_trajectory_prompt(
                    state["agentevals_evaluation_criteria"]
                ),
            )
        eval_result = evaluator(inputs=inputs, outputs=outputs)
        if not eval_result["score"]:
            message = HumanMessage(
                content=f"""
In your last attempt to solve the given task, you took a bad trajectory. Here's why:

<rationale>
{eval_result["comment"]}
</rationale>

Stop and reflect on the original task again, and come up with a new plan to fix these mistakes that takes the above rationale into account.
"""
            )
            # Remove all previous messages except the first one
            # cleared_current_messages = [RemoveMessage(message.id) for message in state["messages"][1:]]
            # return {"messages": cleared_current_messages + [message]}
            return {"messages": [message]}
        return {"messages": []}

    def restart_or_end(state: ReflectionAgentState) -> ReflectionAgentState:
        return "agent" if state["messages"][-1].type == "human" else "__end__"

    reflection_graph = StateGraph(ReflectionAgentState)
    reflection_graph.add_node("agent", graph)
    reflection_graph.add_node("reflect", reflect)
    reflection_graph.add_edge("agent", "reflect")
    reflection_graph.add_conditional_edges("reflect", restart_or_end)

    if evaluator is None:
        reflection_graph.add_node(
            "generate_evaluation_criteria", generate_evaluation_criteria
        )
        reflection_graph.add_edge("__start__", "generate_evaluation_criteria")
        reflection_graph.add_edge("generate_evaluation_criteria", "agent")
    else:
        reflection_graph.add_edge("__start__", "agent")
    return reflection_graph.compile()
