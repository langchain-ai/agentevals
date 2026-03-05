import { BaseMessage } from "@langchain/core/messages";
import { createTrajectoryMatchEvaluator } from "openevals";
import type {
  EvaluatorResult,
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
} from "openevals";

/**
 * @deprecated Use `createTrajectoryMatchEvaluator` from openevals with `trajectoryMatchMode: "superset"` instead.
 */
export async function trajectorySuperset(params: {
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
    trajectoryMatchMode: "superset",
    toolArgsMatchMode: "ignore",
  });
  return evaluator({
    outputs: params.outputs,
    referenceOutputs: params.referenceOutputs,
  });
}
