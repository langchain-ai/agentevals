from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import MessagesState

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import MessageLikeRepresentation
from langchain_core.runnables import Runnable, RunnableLambda

from langchain.chat_models import init_chat_model

from agentevals.trajectory.llm import create_trajectory_llm_as_judge

from openevals.utils import (
    _chat_completion_messages_to_string,
    _normalize_to_openai_messages_list,
)
from openevals.types import ChatCompletionMessage, EvaluatorResult, SimpleEvaluator

from typing import Callable, Literal, Optional, Union


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

This did not meet the required threshold.

<critique>
{eval_result["comment"]}
</critique>

Stop and reflect on the original task again, and think of a revised plan to fix these errors and improve your score.
"""
    )


def _default_criteria_generator(messages: list[MessageLikeRepresentation]) -> str:
    llm = init_chat_model("openai:o3-mini")
    task = _chat_completion_messages_to_string(
        _normalize_to_openai_messages_list(messages)
    )
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
{task}
</task>
""",
            },
        ]
    )
    return res.content


def wrap_agent_with_reflection(
    *,
    agent: Union[CompiledStateGraph, Callable[[dict], dict]],
    evaluators: list[Optional[SimpleEvaluator]] = None,
    evaluator_type: Union[
        Literal["trajectory", "final_output"],
        list[Literal["trajectory", "final_output"]],
    ] = "trajectory",
    criteria_generator: Optional[
        Callable[[list[MessageLikeRepresentation]], str]
    ] = None,
    reflection_response_formatter: Optional[
        Callable[[EvaluatorResult], ChatCompletionMessage]
    ] = None,
    max_reflections: int = 5,
    max_reflections_strategy: Literal["raise", "return"] = "raise",
    evaluator_score_threshold: float = 0.5,
) -> Runnable:
    if criteria_generator is not None and evaluators:
        raise ValueError("Cannot provide both a criteria generator and an evaluator")

    class ReflectionAgentState(
        agent.builder.schema if isinstance(agent, CompiledStateGraph) else MessagesState
    ):
        agentevals_evaluation_criteria: Optional[str]
        reflection_attempts: Optional[int]
        original_input_messages: Optional[list]
        nested_agent_call_params: Optional[dict]

    def generate_evaluation_criteria(
        state: ReflectionAgentState,
    ) -> ReflectionAgentState:
        nonlocal evaluators
        if criteria_generator is None:
            criteria = _default_criteria_generator(state["original_input_messages"])
        else:
            criteria = criteria_generator(state["original_input_messages"])
        if not evaluators:
            evaluators = [
                create_trajectory_llm_as_judge(
                    model="openai:o3-mini",
                    prompt=_generate_default_trajectory_prompt(criteria),
                )
            ]
        return ReflectionAgentState(agentevals_evaluation_criteria=criteria)

    def reflect(state: ReflectionAgentState) -> ReflectionAgentState:
        inputs = state["original_input_messages"]
        for i, evaluator in enumerate(evaluators):
            current_evaluator_type = (
                evaluator_type[i]
                if isinstance(evaluator_type, list)
                else evaluator_type
            )
            outputs = (
                state["messages"]
                if current_evaluator_type == "trajectory"
                else [state["messages"][-1]]
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
    reflection_graph.add_node(
        "store_original_messages",
        lambda state: ReflectionAgentState(original_input_messages=state["messages"]),
    )
    reflection_graph.add_node(
        "agent",
        agent
        if isinstance(agent, CompiledStateGraph)
        else lambda state: agent(
            {"messages": state["messages"], **state["nested_agent_call_params"]}
        ),
    )
    reflection_graph.add_edge("__start__", "store_original_messages")
    reflection_graph.add_node("reflect", reflect)
    reflection_graph.add_edge("agent", "reflect")
    reflection_graph.add_conditional_edges(
        "reflect", restart_or_end, ["agent", "__end__"]
    )

    if not evaluators or criteria_generator is not None:
        reflection_graph.add_node(
            "generate_evaluation_criteria", generate_evaluation_criteria
        )
        reflection_graph.add_edge(
            "store_original_messages", "generate_evaluation_criteria"
        )
        reflection_graph.add_edge("generate_evaluation_criteria", "agent")
    else:
        reflection_graph.add_edge("store_original_messages", "agent")
    wrapped_agent = reflection_graph.compile()
    if isinstance(agent, CompiledStateGraph):
        return wrapped_agent
    else:
        return (
            RunnableLambda(lambda params: {**params, "nested_agent_call_params": params})
            | wrapped_agent
        )
