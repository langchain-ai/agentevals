import { BaseMessage } from "@langchain/core/messages";
import {
  ChatCompletionMessage,
  EvaluatorResult,
  ToolArgsMatchMode,
  ToolArgsMatchOverrides,
} from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _isTrajectorySuperset } from "./utils.js";

export const _scorer = (params: {
  outputs: ChatCompletionMessage[];
  referenceOutputs: ChatCompletionMessage[];
  toolArgsMatchMode: ToolArgsMatchMode;
  toolArgsMatchOverrides?: ToolArgsMatchOverrides;
}): boolean => {
  const isSubset = _isTrajectorySuperset(
    params.referenceOutputs,
    params.outputs,
    params.toolArgsMatchMode,
    params.toolArgsMatchOverrides
  );
  return isSubset;
};

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` with `trajectoryMatchMode: "subset"` instead.
 * Evaluate whether an agent trajectory and called tools is a subset of a reference trajectory and called tools.
 * This means the agent called a subset of the tools specified in the reference trajectory.
 *
 * @param params - The parameters for trajectory subset evaluation
 * @param params.outputs - Actual trajectory the agent followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @param params.reference_outputs - Ideal reference trajectory the agent should have followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
 */
export async function trajectorySubset(params: {
  outputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
  referenceOutputs:
    | ChatCompletionMessage[]
    | BaseMessage[]
    | { messages: (BaseMessage | ChatCompletionMessage)[] };
}): Promise<EvaluatorResult> {
  const { outputs, referenceOutputs } = params;
  const outputsList = _normalizeToOpenAIMessagesList(outputs);
  const referenceOutputsList = _normalizeToOpenAIMessagesList(referenceOutputs);

  return _runEvaluator("trajectory_subset", _scorer, "trajectory_subset", {
    ...params,
    outputs: outputsList,
    referenceOutputs: referenceOutputsList,
    toolArgsMatchMode: "ignore",
  });
}
