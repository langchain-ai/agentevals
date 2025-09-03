import { createLLMAsJudge } from "openevals/llm";

export * from "openevals/types";

// More tolerant version of ChatCompletionMessage that allows missing tool_call_id
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type FlexibleChatCompletionMessage = Record<string, any> &
  (
    | {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        content: any;
        role: "user" | "system" | "developer";
        id?: string;
      }
    | {
        role: "assistant";
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        content: any;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        tool_calls?: any[];
        id?: string;
      }
    | {
        role: "tool";
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        content: any;
        tool_call_id?: string; // Made optional for backward compatibility
        id?: string;
      }
  );

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

export type TrajectoryLLMAsJudgeParams = Partial<
  Omit<Parameters<typeof createLLMAsJudge>[0], "prompt">
> & {
  prompt?: Parameters<typeof createLLMAsJudge>[0]["prompt"];
};

export type ToolArgsMatchMode = "exact" | "ignore" | "subset" | "superset";

export type ToolArgsMatcher = (
  toolCall: Record<string, unknown>,
  referenceToolCall: Record<string, unknown>
) => boolean | Promise<boolean>;

export type ToolArgsMatchOverrides = Record<
  string,
  ToolArgsMatchMode | string[] | ToolArgsMatcher
>;
