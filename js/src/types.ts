import { createLLMAsJudge } from "openevals/llm";
import type { ChatCompletionMessage } from "openevals/types";

export * from "openevals/types";

export type ChatCompletionMessageWithOptionalContent = ChatCompletionMessage & {
  content?: string;
};

// Trajectory extracted from agent
export type GraphTrajectory = {
  inputs?: (Record<string, unknown> | null)[];
  results: Record<string, unknown>[];
  steps: string[][];
};

// Trajectory extracted from a LangGraph thread
export type ExtractedLangGraphThreadTrajectory = {
  inputs: (Record<string, unknown> | null)[][];
  outputs: GraphTrajectory;
};

export type TrajectoryLLMAsJudgeParams = Omit<
  Parameters<typeof createLLMAsJudge>[0],
  "prompt"
> & { prompt?: string };

export type ToolArgsMatchMode = "exact" | "ignore" | "subset" | "superset";

export type ToolArgsMatcher = (
  toolCall: Record<string, unknown>,
  referenceToolCall: Record<string, unknown>
) => boolean | Promise<boolean>;

export type ToolArgsMatchOverrides = Record<
  string,
  ToolArgsMatchMode | string[] | ToolArgsMatcher
>;
