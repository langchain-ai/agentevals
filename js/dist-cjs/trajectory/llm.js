"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createTrajectoryLLMAsJudge = exports.DEFAULT_NO_REF_PROMPT = exports.DEFAULT_REF_COMPARE_PROMPT = void 0;
const llm_1 = require("openevals/llm");
const utils_js_1 = require("../utils.js");
const utils_js_2 = require("../utils.js");
const utils_js_3 = require("./utils.js");
exports.DEFAULT_REF_COMPARE_PROMPT = `You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal trajectory.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is relatively efficient, though it does not need to be perfectly efficient
  - Is semantically equivalent to the provided reference trajectory, if present
</Rubric>

Grade the following trajectory:

<trajectory>
{outputs}
</trajectory>
{inputs}
{reference_outputs}
`;
exports.DEFAULT_NO_REF_PROMPT = `You are an expert data labeler.
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
{inputs}
`;
function _formatInputs(params) {
    const { inputs, outputs, referenceOutputs } = params;
    const normalizedOutputs = (0, utils_js_2._normalizeToOpenAIMessagesList)(outputs);
    const normalizedReferenceOutputs = (0, utils_js_2._normalizeToOpenAIMessagesList)(referenceOutputs ?? []);
    const formattedReferenceOutputs = normalizedReferenceOutputs
        ? `\nUse the following trajectory as an example reference when grading:\n<reference_trajectory>\n${(0, utils_js_3._chatCompletionMessagesToString)(normalizedReferenceOutputs)}\n</reference_trajectory>\n`
        : "";
    const formattedInputs = inputs
        ? `\nThe agent generated the trajectory from the following input:\n<input>\n${JSON.stringify(inputs)}\n</input>\n`
        : "";
    const formattedOutputs = typeof outputs === "object" && !Array.isArray(outputs)
        ? outputs
        : (0, utils_js_3._chatCompletionMessagesToString)(normalizedOutputs);
    return [
        formattedOutputs,
        formattedReferenceOutputs,
        formattedInputs,
    ];
}
/**
 * Creates an evaluator that uses an LLM to judge agent trajectories.
 *
 * @param options - Configuration options
 * @param options.prompt - The evaluation prompt. Can be a string template, LangChain prompt template,
 *                        or callable that returns a list of chat messages. Note that the default prompt
 *                        allows a rubric in addition to the typical "inputs", "outputs", and
 *                        "reference_outputs" parameters.
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
const createTrajectoryLLMAsJudge = ({ prompt = exports.DEFAULT_REF_COMPARE_PROMPT, feedbackKey = "trajectory_accuracy", model, system, judge, continuous = false, choices, useReasoning = true, fewShotExamples, }) => {
    const scorer = (0, llm_1._createLLMAsJudgeScorer)({
        prompt,
        judge,
        model,
        system,
        continuous,
        choices,
        useReasoning,
        fewShotExamples,
    });
    const wrappedEvaluator = async ({ inputs, outputs, referenceOutputs, rubric, ...extra }) => {
        const [formattedOutputs, formattedReferenceOutputs, formattedInputs,] = prompt === exports.DEFAULT_REF_COMPARE_PROMPT || prompt === exports.DEFAULT_NO_REF_PROMPT ? _formatInputs({ inputs, outputs, referenceOutputs }) : [inputs, (0, utils_js_2._normalizeToOpenAIMessagesList)(outputs), (0, utils_js_2._normalizeToOpenAIMessagesList)(referenceOutputs)];
        return (0, utils_js_1._runEvaluator)(`llm_as_${feedbackKey}_judge`, scorer, feedbackKey, {
            outputs: formattedOutputs,
            referenceOutputs: formattedReferenceOutputs,
            inputs: formattedInputs,
            ...extra,
        });
    };
    return wrappedEvaluator;
};
exports.createTrajectoryLLMAsJudge = createTrajectoryLLMAsJudge;
