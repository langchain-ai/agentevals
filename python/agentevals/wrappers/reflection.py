from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import MessageLikeRepresentation, _convert_to_message

from langchain.chat_models import init_chat_model

from agentevals.trajectory.llm import create_trajectory_llm_as_judge
from openevals.types import ChatCompletionMessage, EvaluatorResult, SimpleEvaluator

from typing import Callable, Literal, Optional


def _generate_default_trajectory_prompt(criteria: str) -> str:
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
"""


def _default_reflection_response_formatter(
    eval_result: EvaluatorResult,
) -> ChatCompletionMessage:
    return HumanMessage(
        content=f"""
In your last attempt to solve the given task, you made some errors, and an evaluator measuring "{eval_result["key"]}"
gave your response a score of:

<score>
{eval_result["score"]}
</score>

This did not meet the required threshold. Here's a critique:

<critique>
{eval_result["comment"]}
</critique>

Stop and reflect on the original task again, and think of a revised plan to fix these errors and improve your score.
"""
    )


def _default_criteria_generator(message: MessageLikeRepresentation) -> str:
    llm = init_chat_model("openai:o3-mini")
    msg = _convert_to_message(message)
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
{msg.content}
</task>
""",
            },
        ]
    )
    return res.content


def wrap_agent_with_reflection(
    *,
    graph: CompiledStateGraph,
    evaluator: Optional[SimpleEvaluator] = None,
    evaluator_type: Literal["trajectory", "final_output"] = "trajectory",
    criteria_generator: Optional[Callable] = None,
    reflection_response_formatter: Optional[
        Callable[[EvaluatorResult], ChatCompletionMessage]
    ] = None,
    max_reflections: int = 5,
    max_reflections_strategy: Literal["raise", "return"] = "raise",
    evaluator_score_threshold: float = 0.5,
) -> CompiledStateGraph:
    if criteria_generator is not None and evaluator is not None:
        raise ValueError("Cannot provide both a criteria generator and an evaluator")

    class ReflectionAgentState(graph.builder.schema):
        agentevals_evaluation_criteria: str
        reflection_attempts: int

    def generate_evaluation_criteria(
        state: ReflectionAgentState,
    ) -> ReflectionAgentState:
        nonlocal criteria_generator
        if criteria_generator is None:
            criteria = _default_criteria_generator(state["messages"][-1])
        else:
            criteria = criteria_generator(state["messages"][-1])
        return ReflectionAgentState(agentevals_evaluation_criteria=criteria)

    def reflect(state: ReflectionAgentState) -> ReflectionAgentState:
        nonlocal \
            evaluator, \
            reflection_response_formatter, \
            evaluator_type, \
            max_reflections, \
            max_reflections_strategy
        inputs = state["messages"][0].content
        outputs = (
            state["messages"]
            if evaluator_type == "trajectory"
            else [state["messages"][-1]]
        )
        if evaluator is None:
            evaluator = create_trajectory_llm_as_judge(
                model="openai:o3-mini",
                prompt=_generate_default_trajectory_prompt(
                    state.get("agentevals_evaluation_criteria", "")
                ),
            )
        eval_result = evaluator(
            inputs=inputs,
            outputs=outputs,
            criteria=state.get("agentevals_evaluation_criteria", ""),
        )
        if eval_result["score"] < evaluator_score_threshold:
            if state.get("reflection_attempts", 0) > max_reflections:
                if max_reflections_strategy == "raise":
                    raise ValueError(
                        f"Could not generate a suitable response in {max_reflections} reflections."
                    )
                else:
                    return ReflectionAgentState(
                        messages=[],
                        reflection_attempts=state.get("reflection_attempts", 0) + 1,
                    )
            if reflection_response_formatter is None:
                message = _default_reflection_response_formatter(eval_result)
            else:
                message = reflection_response_formatter(eval_result)
            return ReflectionAgentState(
                messages=[message],
                reflection_attempts=state.get("reflection_attempts", 0) + 1,
            )
        return ReflectionAgentState(
            messages=[],
            reflection_attempts=state.get("reflection_attempts", 0) + 1,
        )

    def restart_or_end(state: ReflectionAgentState) -> str:
        return "agent" if state["messages"][-1].type == "human" else "__end__"

    reflection_graph = StateGraph(ReflectionAgentState)
    reflection_graph.add_node("agent", graph)
    reflection_graph.add_node("reflect", reflect)
    reflection_graph.add_edge("agent", "reflect")
    reflection_graph.add_conditional_edges("reflect", restart_or_end)

    if evaluator is None or criteria_generator is not None:
        reflection_graph.add_node(
            "generate_evaluation_criteria", generate_evaluation_criteria
        )
        reflection_graph.add_edge("__start__", "generate_evaluation_criteria")
        reflection_graph.add_edge("generate_evaluation_criteria", "agent")
    else:
        reflection_graph.add_edge("__start__", "agent")
    return reflection_graph.compile()
