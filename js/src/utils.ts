import { BaseMessage, isBaseMessage } from "@langchain/core/messages";
import { _convertMessagesToOpenAIParams } from "@langchain/openai";
import {
  _runEvaluator as baseRunEvaluator,
  EvaluationResultType,
} from "openevals/utils";
import {
  ChatCompletionMessage,
  FlexibleChatCompletionMessage,
  MultiResultScorerReturnType,
  SingleResultScorerReturnType,
} from "./types.js";

export const _convertToOpenAIMessage = (
  message: BaseMessage | ChatCompletionMessage
): ChatCompletionMessage => {
  if (isBaseMessage(message)) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return _convertMessagesToOpenAIParams([message])[0] as any;
  } else {
    return message;
  }
};

export const _convertToChatCompletionMessage = (
  message: BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage
): ChatCompletionMessage => {
  let converted: FlexibleChatCompletionMessage;

  if (isBaseMessage(message)) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    converted = _convertMessagesToOpenAIParams([message])[0] as any;
  } else {
    converted = message as FlexibleChatCompletionMessage;
  }

  // For tool messages without tool_call_id, generate one for compatibility
  if (converted.role === "tool" && !converted.tool_call_id) {
    converted = {
      ...converted,
      tool_call_id: `generated-${Math.random().toString(36).substring(2)}`,
    };
  }

  return converted as ChatCompletionMessage;
};

export const _normalizeToOpenAIMessagesList = (
  messages?:
    | (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)[]
    | {
        messages: (
          | BaseMessage
          | ChatCompletionMessage
          | FlexibleChatCompletionMessage
        )[];
      }
): ChatCompletionMessage[] => {
  if (!messages) {
    return [];
  }
  let messagesList: (
    | BaseMessage
    | ChatCompletionMessage
    | FlexibleChatCompletionMessage
  )[];
  if (!Array.isArray(messages)) {
    if ("messages" in messages && Array.isArray(messages.messages)) {
      messagesList = messages.messages;
    } else {
      throw new Error(
        `If passing messages as an object, it must contain a "messages" key`
      );
    }
  } else {
    messagesList = messages;
  }
  return messagesList.map(_convertToChatCompletionMessage);
};

export const processScore = (
  _: string,
  value: boolean | number | { score: boolean | number; reasoning?: string }
) => {
  if (typeof value === "object") {
    if (value != null && "score" in value) {
      return [
        value.score,
        "reasoning" in value && typeof value.reasoning === "string"
          ? value.reasoning
          : undefined,
      ] as const;
    } else {
      throw new Error(
        `Expected a dictionary with a "score" key, but got "${JSON.stringify(
          value,
          null,
          2
        )}"`
      );
    }
  }
  return [value] as const;
};

export const _runEvaluator = async <
  T extends Record<string, unknown>,
  O extends
    | SingleResultScorerReturnType
    | MultiResultScorerReturnType
    | Promise<SingleResultScorerReturnType | MultiResultScorerReturnType>,
>(
  runName: string,
  scorer: (params: T) => O,
  feedbackKey: string,
  extra?: T
): Promise<EvaluationResultType<O>> => {
  return baseRunEvaluator(runName, scorer, feedbackKey, extra, "agentevals");
};
