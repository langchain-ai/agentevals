from __future__ import annotations
from openevals.llm import (
    _create_llm_as_judge_scorer,
    _create_async_llm_as_judge_scorer,
    ChatCompletionMessage,
    ModelClient,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    Callable,
    Optional,
    Union,
)
from langchain_core.runnables import Runnable

from agentevals.types import EvaluatorResult, FewShotExample, GraphTrajectory
from agentevals.utils import _run_evaluator, _arun_evaluator

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_REF_COMPARE_PROMPT = """You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal steps in resolving a user queries.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is relatively efficient, though it does not need to be perfectly efficient
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
"""


def _format_thread(
    inputs: list,
    outputs: GraphTrajectory,
) -> str:
    formatted_thread = ""
    for input, result, step in zip(inputs, outputs["results"], outputs["steps"]):
        formatted_thread += f"\n<input>\n{input}\n</input>\n" if input else ""
        formatted_thread += f"\n<trajectory>\n{step}\n</trajectory>\n"
        formatted_thread += f"\n<result>\n{result}\n</result>\n"
    return formatted_thread


def _format_inputs(
    inputs: list | dict,
    outputs: GraphTrajectory,
    reference_outputs: Optional[GraphTrajectory],
) -> tuple[str, str]:
    if isinstance(inputs, dict):
        if "inputs" not in inputs:
            raise ValueError("inputs must be a list or a dict with an 'inputs' key")
        inputs = inputs["inputs"]
    if len(inputs) != len(outputs["results"]):
        raise ValueError(
            "Provided `inputs` and `results` within provided `outputs` must have the same length"
        )
    if len(inputs) != len(outputs["steps"]):
        raise ValueError(
            "Provided `inputs` and `steps` within provided `outputs` must have the same length"
        )
    formatted_thread = _format_thread(inputs, outputs)  # type: ignore
    if reference_outputs:
        formatted_reference_outputs = f"\nUse the following trajectory as an example reference when grading:\n<reference_thread>\n{_format_thread(reference_outputs['inputs'], reference_outputs)}\n</reference_thread>\n"
    else:
        formatted_reference_outputs = ""
    return (
        formatted_thread,
        formatted_reference_outputs,
    )


def create_graph_trajectory_llm_as_judge(
    *,
    prompt: str
    | Runnable
    | Callable[..., list[ChatCompletionMessage]] = DEFAULT_REF_COMPARE_PROMPT,
    model: Optional[str] = None,
    feedback_key: str = "graph_trajectory_accuracy",
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that uses an LLM to judge agent trajectories.

    Args:
        prompt: The evaluation prompt, can be a string template, LangChain prompt template, or callable
            that returns a list of chat messages. Note that the default prompt allows a rubric
            in addition to the typical "inputs", "outputs", and "reference_outputs" parameters.
        feedback_key: Key used to store the evaluation result, defaults to "graph_trajectory_accuracy".
        judge: The LLM used for evaluation. Can be an OpenAI client),
            or a LangChain chat model. If an OpenAI client, must specify "model" as well.
            If omitted, "model" will be used to instantiate a LangChain model instance
            by model string.
        model: Model identifier to use. If "judge" is an OpenAI client,
            this argument should be a model name directly. If "judge" is omitted, must be a valid
            LangChain model identifier. See `init_chat_model` docs for more details:
            https://python.langchain.com/docs/how_to/chat_models_universal_init/.
        system: Optional system message to prepend to the prompt.
        continuous: If True, score will be a float between 0 and 1. If False, score will be boolean. Defaults to False.
        choices: Optional list of specific float values the score must be chosen from.
        use_reasoning: If True, includes explanation for the score in the output. Defaults to True.
        few_shot_examples: Optional list of example evaluations to append to the prompt.

    Returns:
        SimpleEvaluator: A function that evaluates agent trajectories using the configured LLM judge.
    """
    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    def _wrapped_evaluator(
        *,
        inputs: Optional[Union[dict, list]] = None,
        outputs: GraphTrajectory,
        reference_outputs: Optional[GraphTrajectory] = None,
        **kwargs,
    ) -> EvaluatorResult:
        (
            formatted_thread,
            formatted_reference_outputs,
        ) = _format_inputs(inputs, outputs, reference_outputs)
        return _run_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs=outputs,
            inputs=inputs,
            thread=formatted_thread,
            reference_outputs=formatted_reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_graph_trajectory_llm_as_judge(
    *,
    prompt: str
    | Runnable
    | Callable[..., list[ChatCompletionMessage]] = DEFAULT_REF_COMPARE_PROMPT,
    model: Optional[str] = None,
    feedback_key: str = "graph_trajectory_accuracy",
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleAsyncEvaluator:
    """Creates an evaluator that uses an LLM to judge agent trajectories.

    Args:
        prompt: The evaluation prompt, can be a string template, LangChain prompt template, or callable
            that returns a list of chat messages. Note that the default prompt allows a rubric
            in addition to the typical "inputs", "outputs", and "reference_outputs" parameters.
        feedback_key: Key used to store the evaluation result, defaults to "graph_trajectory_accuracy".
        judge: The LLM used for evaluation. Can be an OpenAI client),
            or a LangChain chat model. If an OpenAI client, must specify "model" as well.
            If omitted, "model" will be used to instantiate a LangChain model instance
            by model string.
        model: Model identifier to use. If "judge" is an OpenAI client,
            this argument should be a model name directly. If "judge" is omitted, must be a valid
            LangChain model identifier. See `init_chat_model` docs for more details:
            https://python.langchain.com/docs/how_to/chat_models_universal_init/.
        system: Optional system message to prepend to the prompt.
        continuous: If True, score will be a float between 0 and 1. If False, score will be boolean. Defaults to False.
        choices: Optional list of specific float values the score must be chosen from.
        use_reasoning: If True, includes explanation for the score in the output. Defaults to True.
        few_shot_examples: Optional list of example evaluations to append to the prompt.

    Returns:
        SimpleAsyncEvaluator: A function that evaluates agent trajectories using the configured LLM judge.
    """
    scorer = _create_async_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    async def _wrapped_evaluator(
        *,
        inputs: Optional[Union[dict, list]] = None,
        outputs: GraphTrajectory,
        reference_outputs: Optional[GraphTrajectory] = None,
        **kwargs,
    ) -> EvaluatorResult:
        (
            formatted_thread,
            formatted_reference_outputs,
        ) = _format_inputs(inputs, outputs, reference_outputs)
        return await _arun_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs="",
            thread=formatted_thread,
            reference_outputs=formatted_reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator
