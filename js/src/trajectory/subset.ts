import { BaseMessage } from "@langchain/core/messages";
import { createTrajectoryMatchEvaluator } from "openevals";
import type {
  EvaluatorResult,
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
} from "openevals";

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` from openevals with `trajectoryMatchMode: "subset"` instead.
 */
export async function trajectorySubset(params: {
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
  const evaluator = createTrajectoryMatchEvaluator({
    trajectoryMatchMode: "subset",
    toolArgsMatchMode: "ignore",
  });
  return evaluator({
    outputs: params.outputs,
    referenceOutputs: params.referenceOutputs,
  });
}
