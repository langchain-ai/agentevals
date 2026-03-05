import { BaseMessage } from "@langchain/core/messages";
import { createTrajectoryMatchEvaluator } from "openevals";
import type {
  EvaluatorResult,
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
} from "openevals";

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` from openevals with `trajectoryMatchMode: "strict"` instead.
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
  toolCallArgsExactMatch?: boolean;
}): Promise<EvaluatorResult> {
  const evaluator = createTrajectoryMatchEvaluator({
    trajectoryMatchMode: "strict",
    toolArgsMatchMode:
      params.toolCallArgsExactMatch !== false ? "exact" : "ignore",
  });
  return evaluator({
    outputs: params.outputs,
    referenceOutputs: params.referenceOutputs,
  });
}
