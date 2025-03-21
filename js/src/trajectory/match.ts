import { BaseMessage } from "@langchain/core/messages";
import {
  ChatCompletionMessage,
  ToolArgsMatchMode,
  ToolArgsMatchOverrides,
} from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _scorer as trajectoryStrictScorer } from "./strict.js";
import { _scorer as trajectoryUnorderedScorer } from "./unordered.js";
import { _scorer as trajectorySubsetScorer } from "./subset.js";
import { _scorer as trajectorySuperstScorer } from "./superset.js";

export type TrajectoryMatchMode =
  | "strict"
  | "unordered"
  | "subset"
  | "superset";

export function createTrajectoryMatchEvaluator({
  trajectoryMatchMode = "strict",
  toolArgsMatchMode = "exact",
  toolArgsMatchOverrides,
}: {
  trajectoryMatchMode?: TrajectoryMatchMode;
  toolArgsMatchMode?: ToolArgsMatchMode;
  toolArgsMatchOverrides?: ToolArgsMatchOverrides;
}) {
  let scorer: (params: {
    outputs: ChatCompletionMessage[];
    referenceOutputs: ChatCompletionMessage[];
    toolArgsMatchMode: ToolArgsMatchMode;
    toolArgsMatchOverrides?: ToolArgsMatchOverrides;
  }) => boolean;
  switch (trajectoryMatchMode) {
    case "strict":
      scorer = trajectoryStrictScorer;
      break;
    case "unordered":
      scorer = trajectoryUnorderedScorer;
      break;
    case "subset":
      scorer = trajectorySubsetScorer;
      break;
    case "superset":
      scorer = trajectorySuperstScorer;
      break;
    default:
      throw new Error(`Invalid trajectory match type: ${trajectoryMatchMode}`);
  }

  return async function _wrappedEvaluator({
    outputs,
    referenceOutputs,
    ...extra
  }: {
    outputs:
      | ChatCompletionMessage[]
      | BaseMessage[]
      | { messages: (BaseMessage | ChatCompletionMessage)[] };
    referenceOutputs:
      | ChatCompletionMessage[]
      | BaseMessage[]
      | { messages: (BaseMessage | ChatCompletionMessage)[] };
    [key: string]: unknown;
  }) {
    const normalizedOutputs = _normalizeToOpenAIMessagesList(outputs);
    const normalizedReferenceOutputs =
      _normalizeToOpenAIMessagesList(referenceOutputs);

    return _runEvaluator(
      `trajectory_${trajectoryMatchMode}_match`,
      scorer,
      `trajectory_${trajectoryMatchMode}_match`,
      {
        outputs: normalizedOutputs,
        referenceOutputs: normalizedReferenceOutputs,
        toolArgsMatchMode,
        toolArgsMatchOverrides,
        ...extra,
      }
    );
  };
}
