import { BaseMessage } from "@langchain/core/messages";
import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
  EvaluatorResult,
  ToolArgsMatchMode,
  ToolArgsMatchOverrides,
} from "../types.js";
import { _normalizeToOpenAIMessagesList, _runEvaluator } from "../utils.js";
import { _getMatcherForToolName } from "./utils.js";

export async function _scorer(params: {
  outputs: ChatCompletionMessage[];
  referenceOutputs: ChatCompletionMessage[];
  toolArgsMatchMode: ToolArgsMatchMode;
  toolArgsMatchOverrides?: ToolArgsMatchOverrides;
}): Promise<boolean | [false, string]> {
  const {
    outputs,
    referenceOutputs,
    toolArgsMatchMode,
    toolArgsMatchOverrides,
  } = params;
  const normalizedOutputs = outputs;
  const normalizedReferenceOutputs = referenceOutputs;

  if (!normalizedOutputs || !normalizedReferenceOutputs) {
    throw new Error(
      "Strict trajectory match requires both outputs and reference_outputs"
    );
  }

  if (normalizedOutputs.length !== normalizedReferenceOutputs.length) {
    return [
      false,
      `Trajectory length mismatch: output trajectory has ${normalizedOutputs.length} messages, reference trajectory has ${normalizedReferenceOutputs.length} messages.`,
    ];
  }

  for (let i = 0; i < normalizedOutputs.length; i++) {
    const output = normalizedOutputs[i];
    const referenceOutput = normalizedReferenceOutputs[i];

    if (output.role !== referenceOutput.role) {
      return [
        false,
        `Role mismatch at step ${i}: expected ${referenceOutput.role} but got ${output.role}`,
      ];
    }

    const outputHasToolCalls = output.tool_calls != null;
    const referenceHasToolCalls = referenceOutput.tool_calls != null;

    if (outputHasToolCalls !== referenceHasToolCalls) {
      return [
        false,
        `Tool call mismatch at step ${i}: expected ${referenceHasToolCalls ? "tool calls" : "no tool calls"} but got ${outputHasToolCalls ? "tool calls" : "no tool calls"}`,
      ];
    }

    if (outputHasToolCalls) {
      if (output.tool_calls!.length !== referenceOutput.tool_calls!.length) {
        return [
          false,
          `Tool call count mismatch at step ${i}: expected ${referenceOutput.tool_calls!.length} but got ${output.tool_calls!.length}`,
        ];
      }
      const referenceCalls = referenceOutput.tool_calls ?? [];
      const seen = new Array(referenceCalls.length).fill(false);

      for (const outputCall of output.tool_calls ?? []) {
        let foundMatch = false;
        for (let i = 0; i < referenceCalls.length; i++) {
          const referenceCall = referenceCalls[i];
          if (
            !seen[i] &&
            outputCall.function?.name === referenceCall.function?.name
          ) {
            const matcher = _getMatcherForToolName(
              outputCall.function?.name ?? "",
              toolArgsMatchMode,
              toolArgsMatchOverrides
            );
            if (
              await matcher(
                JSON.parse(outputCall.function?.arguments ?? "{}"),
                JSON.parse(referenceCall.function?.arguments ?? "{}")
              )
            ) {
              foundMatch = true;
              seen[i] = true;
              break;
            }
          }
        }
        if (!foundMatch) {
          return [
            false,
            `Tool call mismatch at step ${i}: no matching tool call found for ${outputCall.function?.name} with arguments ${outputCall.function?.arguments}`,
          ];
        }
      }
    }
  }

  return true;
}

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` with `trajectoryMatchMode: "strict"` instead.
 * Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
 * This means that at each step, the agent called the same tools in the same order as specified in the reference trajectory.
 *
 * @param outputs - Actual trajectory the agent followed. May be a list of OpenAI messages,
 *                 a list of LangChain messages, or a dictionary containing a "messages" key with one of the above.
 * @param referenceOutputs - Ideal reference trajectory the agent should have followed. May be a list of OpenAI messages,
 *                          a list of LangChain messages, or a dictionary containing a "messages" key with one of the above.
 * @param toolCallArgsExactMatch - Whether to require exact matches for tool call arguments
 * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
 */
export async function trajectoryStrictMatch(params: {
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
  referenceOutputs:
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
  toolCallArgsExactMatch: boolean;
}): Promise<EvaluatorResult> {
  const normalizedOutputs = _normalizeToOpenAIMessagesList(params.outputs);
  const normalizedReferenceOutputs = _normalizeToOpenAIMessagesList(
    params.referenceOutputs
  );

  return _runEvaluator(
    "trajectory_strict_match",
    _scorer,
    "trajectory_strict_match",
    {
      outputs: normalizedOutputs,
      referenceOutputs: normalizedReferenceOutputs,
      toolArgsMatchMode: params.toolCallArgsExactMatch ? "exact" : "ignore",
    }
  );
}
