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

/**
 * Creates an evaluator that compares trajectories between model outputs and reference outputs.
 *
 * @param options - The configuration options
 * @param options.trajectoryMatchMode - The mode for matching trajectories:
 *   - `"strict"`: Requires exact match in order and content
 *   - `"unordered"`: Allows matching in any order
 *   - `"subset"`: Accepts if output trajectory is a subset of reference
 *   - `"superset"`: Accepts if output trajectory is a superset of reference
 * @param options.toolArgsMatchMode - Mode for matching tool arguments ("exact" by default, can be "ignore")
 * @param options.toolArgsMatchOverrides - Object containing custom overrides for tool argument matching.
 *   Each key should be a tool name, and each value should be either a match mode or a matcher function.
 *   Matchers should be a function that takes two sets of tool call args and returns whether they are equal.
 *
 * @returns An async function that evaluates trajectory matches between outputs and references.
 *   The returned evaluator accepts:
 *   - outputs: List of messages or dict representing the model output trajectory
 *   - referenceOutputs: List of messages or dict representing the reference trajectory
 *   - Additional arguments passed to the underlying evaluator
 *
 * @example
 * ```typescript
 * const matcher = (
 *   outputToolCallArgs: Record<string, any>,
 *   referenceToolCallArgs: Record<string, any>
 * ): boolean => {
 *   const outputArgs = (outputToolCallArgs.query ?? "").toLowerCase();
 *   const referenceArgs = (referenceToolCallArgs.query ?? "").toLowerCase();
 *   return outputArgs === referenceArgs;
 * };
 *
 * const evaluator = createAsyncTrajectoryMatchEvaluator({
 *   trajectoryMatchMode: "strict",
 *   toolArgsMatchMode: "exact",
 *   toolArgsMatchOverrides: {
 *     myToolName: matcher,
 *   },
 * });
 *
 * const result = await evaluator({
 *   outputs: [...],
 *   referenceOutputs: [...],
 * });
 * ```
 */
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
  }) => boolean | Promise<boolean>;
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
