import { BaseMessage } from "@langchain/core/messages";
import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
} from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _extractToolCalls } from "./utils.js";

function _scoreToolPermissions(
  outputs: ChatCompletionMessage[],
  allowedTools: Set<string> | null,
  deniedTools: Set<string>
): number {
  const toolCalls = _extractToolCalls(outputs);
  if (toolCalls.length === 0) {
    // No tools were called, so no permission boundary could be violated.
    return 1;
  }
  let unauthorized = 0;
  for (const call of toolCalls) {
    const name = call.name;
    if (deniedTools.has(name)) {
      unauthorized += 1;
    } else if (allowedTools !== null && !allowedTools.has(name)) {
      unauthorized += 1;
    }
  }
  return (toolCalls.length - unauthorized) / toolCalls.length;
}

/**
 * Creates an evaluator that checks whether an agent only called tools it was authorized to,
 * based on a permission policy (independent of any reference trajectory).
 *
 * Unlike `createTrajectoryMatchEvaluator`, which compares called tools against a reference
 * trajectory, this evaluator enforces least privilege: it flags any tool call outside the
 * granted policy, regardless of whether the task was completed.
 *
 * @param options - The configuration options
 * @param options.allowedTools - Allowlist of permitted tool names. If provided, any called
 *   tool not in this list counts as unauthorized (least privilege).
 * @param options.deniedTools - Denylist of forbidden tool names. Any called tool in this list
 *   counts as unauthorized. A denial always takes precedence over an allow.
 *
 * @returns An async function that accepts `outputs` (the agent trajectory) and returns an
 *   EvaluatorResult whose score is the fraction of tool calls that were authorized
 *   (1 when no tools were called).
 *
 * @example
 * ```typescript
 * const evaluator = createTrajectoryToolPermissionEvaluator({
 *   allowedTools: ["search_kb", "reply_to_customer"],
 *   deniedTools: ["issue_refund"],
 * });
 * const result = await evaluator({ outputs: [...] });
 * ```
 */
export function createTrajectoryToolPermissionEvaluator({
  allowedTools,
  deniedTools,
}: {
  allowedTools?: string[];
  deniedTools?: string[];
}) {
  if (allowedTools == null && deniedTools == null) {
    throw new Error(
      "createTrajectoryToolPermissionEvaluator requires at least one of " +
        "`allowedTools` (an allowlist) or `deniedTools` (a denylist)."
    );
  }
  const allowed = allowedTools != null ? new Set(allowedTools) : null;
  const denied = new Set(deniedTools ?? []);

  const scorer = (params: {
    outputs: ChatCompletionMessage[];
    [key: string]: unknown;
  }): number => _scoreToolPermissions(params.outputs, allowed, denied);

  return async function _wrappedEvaluator({
    outputs,
    ...extra
  }: {
    outputs:
      | ChatCompletionMessage[]
      | FlexibleChatCompletionMessage[]
      | BaseMessage[]
      | {
          messages: (
            | BaseMessage
            | ChatCompletionMessage
            | FlexibleChatCompletionMessage
          )[];
        };
    [key: string]: unknown;
  }) {
    const normalizedOutputs = _normalizeToOpenAIMessagesList(outputs);
    return _runEvaluator(
      "trajectory_tool_permission",
      scorer,
      "trajectory_tool_permission",
      {
        outputs: normalizedOutputs,
        ...extra,
      }
    );
  };
}
