import { BaseMessage } from "@langchain/core/messages";
import { _createLLMAsJudgeScorer } from "openevals/llm";

import { _runEvaluator, _normalizeToOpenAIMessagesList } from "../utils.js";
import { _chatCompletionMessagesToString } from "./utils.js";
import {
  ChatCompletionMessage,
  EvaluatorResult,
  TrajectoryLLMAsJudgeParams,
} from "../types.js";

export const TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE = `You are an expert data labeler.
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
`;

export const TRAJECTORY_ACCURACY_PROMPT = `You are an expert data labeler.
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
</trajectory>`;

function _formatInputs(params: {
  outputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
  referenceOutputs?:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
}): [string, string] {
  const { outputs, referenceOutputs } = params;
  const normalizedOutputs = _normalizeToOpenAIMessagesList(outputs);
  const normalizedReferenceOutputs = _normalizeToOpenAIMessagesList(
    referenceOutputs ?? []
  );

  const formattedReferenceOutputs = normalizedReferenceOutputs
    ? _chatCompletionMessagesToString(normalizedReferenceOutputs)
    : "";

  const formattedOutputs = _chatCompletionMessagesToString(normalizedOutputs);

  return [formattedOutputs, formattedReferenceOutputs];
}

/**
 * Creates an evaluator that uses an LLM to judge agent trajectories.
 *
 * @param options - Configuration options
 * @param options.prompt - The evaluation prompt. Can be a string template, LangChain prompt template,
 *                        or callable that returns a list of chat messages.
 * @param options.feedbackKey - Key used to store the evaluation result. Defaults to "trajectory_accuracy".
 * @param options.model - Model identifier to use. If judge is an OpenAI client,
 *                       this should be a model name directly. If judge is omitted, must be a valid
 *                       LangChain model identifier.
 * @param options.system - Optional system message to prepend to the prompt.
 * @param options.judge - The LLM used for evaluation. Can be an OpenAI client or a LangChainLikeModel.
 *                       If an OpenAI client, must specify "model" as well. If omitted, "model" will be
 *                       used to instantiate a LangChain model instance by model string.
 * @param options.continuous - If true, score will be a float between 0 and 1. If false, score will be boolean.
 *                           Defaults to false.
 * @param options.choices - Optional list of specific float values the score must be chosen from.
 * @param options.useReasoning - If true, includes explanation for the score in the output. Defaults to true.
 * @param options.fewShotExamples - Optional list of example evaluations to append to the prompt.
 * @returns A function that evaluates agent trajectories using the configured LLM judge.
 */
export const createTrajectoryLLMAsJudge = ({
  prompt = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
  feedbackKey = "trajectory_accuracy",
  model,
  system,
  judge,
  continuous = false,
  choices,
  useReasoning = true,
  fewShotExamples,
}: TrajectoryLLMAsJudgeParams) => {
  const scorer = _createLLMAsJudgeScorer({
    prompt,
    judge,
    model,
    system,
    continuous,
    choices,
    useReasoning,
    fewShotExamples,
  });

  const wrappedEvaluator = async ({
    inputs,
    outputs,
    referenceOutputs,
    ...extra
  }: {
    outputs:
      | ChatCompletionMessage[]
      | BaseMessage[]
      | { messages: (BaseMessage | ChatCompletionMessage)[] };
    referenceOutputs?:
      | ChatCompletionMessage[]
      | BaseMessage[]
      | { messages: (BaseMessage | ChatCompletionMessage)[] };
    [key: string]: unknown;
  }): Promise<EvaluatorResult> => {
    const [formattedOutputs, formattedReferenceOutputs] = _formatInputs({
      outputs,
      referenceOutputs,
    });

    return _runEvaluator(`llm_as_${feedbackKey}_judge`, scorer, feedbackKey, {
      inputs,
      outputs: formattedOutputs,
      referenceOutputs: formattedReferenceOutputs,
      ...extra,
    });
  };
  return wrappedEvaluator;
};
