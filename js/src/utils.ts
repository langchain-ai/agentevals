import type { BaseMessage } from "@langchain/core/messages";
import { isBaseMessage } from "@langchain/core/messages";
import * as openAIImports from "@langchain/openai";
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

type NormalizeToOpenAIMessagesListFunction = (
  messages?:
    | (BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage)[]
    | {
        messages: (
          | BaseMessage
          | ChatCompletionMessage
          | FlexibleChatCompletionMessage
        )[];
      }
) => ChatCompletionMessage[];

const {
  // @ts-expect-error Shim for older versions of @langchain/openai
  _convertMessagesToOpenAIParams,
  convertMessagesToCompletionsMessageParams,
} = openAIImports;

function _convertMessagesShim(message: BaseMessage) {
  if (typeof _convertMessagesToOpenAIParams === "function") {
    return _convertMessagesToOpenAIParams([
      message,
    ])[0] as ChatCompletionMessage;
  }
  return convertMessagesToCompletionsMessageParams({
    messages: [message],
  })[0] as ChatCompletionMessage;
}

export const _convertToOpenAIMessage = (
  message: BaseMessage | ChatCompletionMessage
): ChatCompletionMessage => {
  if (isBaseMessage(message)) {
    const converted = _convertMessagesShim(message);
    if (message.id && !converted.id) {
      converted.id = message.id;
    }
    return converted;
  } else {
    return message;
  }
};

export const _convertToChatCompletionMessage = (
  message: BaseMessage | ChatCompletionMessage | FlexibleChatCompletionMessage
): ChatCompletionMessage => {
  let converted: FlexibleChatCompletionMessage;

  if (isBaseMessage(message)) {
    converted = _convertMessagesShim(message);
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

export const _normalizeToOpenAIMessagesList: NormalizeToOpenAIMessagesListFunction =
  (messages) => {
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
