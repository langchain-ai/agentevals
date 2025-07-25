import { BaseMessage } from "@langchain/core/messages";
import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
  EvaluatorResult,
  ToolArgsMatchMode,
  ToolArgsMatchOverrides,
} from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _isTrajectorySuperset } from "./utils.js";

export const _scorer = async (params: {
  outputs: ChatCompletionMessage[];
  referenceOutputs: ChatCompletionMessage[];
  toolArgsMatchMode: ToolArgsMatchMode;
  toolArgsMatchOverrides?: ToolArgsMatchOverrides;
}): Promise<boolean> => {
  const isUnorderedMatch =
    (await _isTrajectorySuperset(
      params.outputs,
      params.referenceOutputs,
      params.toolArgsMatchMode,
      params.toolArgsMatchOverrides
    )) &&
    (await _isTrajectorySuperset(
      params.referenceOutputs,
      params.outputs,
      params.toolArgsMatchMode,
      params.toolArgsMatchOverrides
    ));
  return isUnorderedMatch;
};

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` with `trajectoryMatchMode: "unordered"` instead.
 * Evaluate whether an input agent trajectory and called tools contains all the tools used in a reference trajectory.
 * This accounts for some differences in an LLM's reasoning process in a case-by-case basis.
 *
 * @param params - The parameters for trajectory unordered match evaluation
 * @param params.outputs - Actual trajectory the agent followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @param params.reference_outputs - Ideal reference trajectory the agent should have followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
 */
export async function trajectoryUnorderedMatch(params: {
  outputs:
    | FlexibleChatCompletionMessage[]
    | BaseMessage[]
    | {
        messages: (
          | BaseMessage
          | ChatCompletionMessage
          | FlexibleChatCompletionMessage
        )[];
      };
  referenceOutputs:
    | FlexibleChatCompletionMessage[]
    | BaseMessage[]
    | {
        messages: (
          | BaseMessage
          | ChatCompletionMessage
          | FlexibleChatCompletionMessage
        )[];
      };
}): Promise<EvaluatorResult> {
  const { outputs, referenceOutputs } = params;
  const outputsList = _normalizeToOpenAIMessagesList(outputs);
  const referenceOutputsList = _normalizeToOpenAIMessagesList(referenceOutputs);

  return _runEvaluator(
    "trajectory_unordered_match",
    _scorer,
    "trajectory_unordered_match",
    {
      ...params,
      outputs: outputsList,
      referenceOutputs: referenceOutputsList,
      toolArgsMatchMode: "ignore",
    }
  );
}
