import { _createLLMAsJudgeScorer } from "openevals/llm";

import { _runEvaluator } from "../utils.js";
import type { GraphTrajectory, TrajectoryLLMAsJudgeParams } from "../types.js";

export const GRAPH_TRAJECTORY_ACCURACY_PROMPT = `You are an expert data labeler.
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
  interrupting to await additional data from another source ("human-in-the-loop").
  
  Steps containing a colon represent steps within subagents (e.g. "graph:step_name").
</Instructions>

<thread>
{thread}
</thread>

{reference_outputs}
`;

function _formatThread(
  inputs: (string | Record<string, unknown> | null)[],
  outputs: GraphTrajectory
): string {
  let formattedThread = "";
  const zippedData = inputs.map((input, i) => ({
    input: JSON.stringify(input ?? ""),
    result: JSON.stringify(outputs.results[i]),
    step: JSON.stringify(outputs.steps[i]),
  }));

  for (const { input, result, step } of zippedData) {
    formattedThread += input ? `\n<input>\n${input}\n</input>\n` : "";
    formattedThread += `\n<trajectory>\n${step}\n</trajectory>\n`;
    formattedThread += `\n<result>\n${result}\n</result>\n`;
  }
  return formattedThread;
}

function _formatInputs(
  inputs:
    | (string | Record<string, unknown> | null)[]
    | { inputs: (string | Record<string, unknown> | null)[] },
  outputs: GraphTrajectory,
  referenceOutputs?: GraphTrajectory
) {
  let processedInputs: (string | Record<string, unknown> | null)[];

  if (Array.isArray(inputs)) {
    processedInputs = inputs;
  } else {
    if (!("inputs" in inputs)) {
      throw new Error(
        "inputs must be an array or an object with an 'inputs' key"
      );
    }
    processedInputs = inputs.inputs;
  }

  if (processedInputs.length !== outputs.results.length) {
    throw new Error(
      "Provided `inputs` and `results` within provided `outputs` must have the same length"
    );
  }
  if (processedInputs.length !== outputs.steps.length) {
    throw new Error(
      "Provided `inputs` and `steps` within provided `outputs` must have the same length"
    );
  }

  const formattedThread = _formatThread(processedInputs, outputs);
  const formattedReferenceOutputs = referenceOutputs
    ? `\nUse the following trajectory as an example reference when grading:\n<reference_thread>\n${_formatThread(referenceOutputs.inputs ?? [], referenceOutputs)}\n</reference_thread>\n`
    : "";

  return {
    formattedThread,
    formattedReferenceOutputs,
  };
}

/**
 * Creates an evaluator that uses an LLM to judge agent trajectories.
 * @param options Configuration options
 * @param [options.prompt] - The evaluation prompt. Can be a string template,
 *        LangChain prompt template, or callable that returns a list of chat messages. Note that the default prompt allows a rubric
 *        in addition to the typical "inputs", "outputs", and "reference_outputs" parameters.
 * @param [options.feedbackKey="graph_trajectory_accuracy"] - Key used to store the evaluation result
 * @param [options.judge] - The LLM used for evaluation. Can be an OpenAI client
 *        or a LangChain chat model. If an OpenAI client, must specify "model" as well.
 *        If omitted, "model" will be used to instantiate a LangChain model instance by model string.
 * @param [options.model] - Model identifier to use. If "judge" is an OpenAI client,
 *        this argument should be a model name directly. If "judge" is omitted, must be a valid
 *        LangChain model identifier. See `init_chat_model` docs for more details:
 *        https://python.langchain.com/docs/how_to/chat_models_universal_init/
 * @param [options.continuous=false] - If true, score will be a float between 0 and 1. If false, score will be boolean.
 * @param [options.choices] - Optional list of specific float values the score must be chosen from
 * @param [options.useReasoning=true] - If true, includes explanation for the score in the output
 * @param [options.fewShotExamples] - Optional list of example evaluations to append to the prompt
 * @returns A function that evaluates agent trajectories using the configured LLM judge
 */
export const createGraphTrajectoryLLMAsJudge = ({
  prompt = GRAPH_TRAJECTORY_ACCURACY_PROMPT,
  model,
  feedbackKey = "graph_trajectory_accuracy",
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
    continuous,
    choices,
    useReasoning,
    fewShotExamples,
  });

  const _wrappedEvaluator = async ({
    inputs,
    outputs,
    referenceOutputs,
    ...extra
  }: {
    inputs:
      | (string | Record<string, unknown> | null)[]
      | { inputs: (string | Record<string, unknown> | null)[] };
    outputs: GraphTrajectory;
    referenceOutputs?: GraphTrajectory;
    [key: string]: unknown;
  }) => {
    const { formattedThread, formattedReferenceOutputs } = _formatInputs(
      inputs,
      outputs,
      referenceOutputs
    );
    return _runEvaluator(`llm_as_${feedbackKey}_judge`, scorer, feedbackKey, {
      outputs,
      inputs,
      thread: formattedThread,
      referenceOutputs: formattedReferenceOutputs,
      ...extra,
    });
  };
  return _wrappedEvaluator;
};
