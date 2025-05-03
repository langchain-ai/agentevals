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
from openevals.utils import (
    _chat_completion_messages_to_string,
    _normalize_to_openai_messages_list,
)
from agentevals.types import FewShotExample
from agentevals.utils import _run_evaluator, _arun_evaluator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from typing import TYPE_CHECKING


TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE = """You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal trajectory.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is relatively efficient, though it does not need to be perfectly efficient
  - Is semantically equivalent to the provided reference trajectory
</Rubric>

Based on the following reference trajectory:

<reference_trajectory>
{reference_outputs}
</reference_trajectory>

Grade this actual trajectory:

<trajectory>
{outputs}
</trajectory>
"""

TRAJECTORY_ACCURACY_PROMPT = """You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal trajectory.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is relatively efficient, though it does not need to be perfectly efficient
</Rubric>

First, try to understand the goal of the trajectory by looking at the input
(if the input is not present try to infer it from the content of the first message),
as well as the output of the final message. Once you understand the goal, grade the trajectory
as it relates to achieving that goal.

Grade the following trajectory:

<trajectory>
{outputs}
</trajectory>
"""

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _format_inputs(
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Optional[
        Union[list[ChatCompletionMessage], list[BaseMessage], dict]
    ],
) -> tuple[str, str]:
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
    if reference_outputs:
        formatted_reference_outputs = _chat_completion_messages_to_string(
            reference_outputs
        )
    else:
        formatted_reference_outputs = ""
    formatted_outputs = _chat_completion_messages_to_string(outputs)
    return (
        formatted_outputs,
        formatted_reference_outputs,
    )


def create_trajectory_llm_as_judge(
    *,
    prompt: str
    | Runnable
    | Callable[
        ..., list[ChatCompletionMessage]
    ] = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    model: Optional[str] = None,
    feedback_key: str = "trajectory_accuracy",
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
        feedback_key: Key used to store the evaluation result, defaults to "trajectory_accuracy".
        judge: The LLM used for evaluation. Can be an OpenAI client
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
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Optional[
            Union[list[ChatCompletionMessage], list[BaseMessage], dict]
        ] = None,
        **kwargs,
    ):
        (
            formatted_outputs,
            formatted_reference_outputs,
        ) = _format_inputs(outputs, reference_outputs)
        return _run_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs=formatted_outputs,
            reference_outputs=formatted_reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_trajectory_llm_as_judge(
    *,
    prompt: str
    | Runnable
    | Callable[
        ..., list[ChatCompletionMessage]
    ] = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    model: Optional[str] = None,
    feedback_key: str = "trajectory_accuracy",
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
        feedback_key: Key used to store the evaluation result, defaults to "trajectory_accuracy".
        judge: The LLM used for evaluation. Can be an OpenAI client
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
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Optional[
            Union[list[ChatCompletionMessage], list[BaseMessage], dict]
        ] = None,
        **kwargs,
    ):
        (
            formatted_outputs,
            formatted_reference_outputs,
        ) = _format_inputs(outputs, reference_outputs)
        return await _arun_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs=formatted_outputs,
            reference_outputs=formatted_reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator
