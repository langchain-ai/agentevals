import { createLLMAsJudge } from "openevals/llm";

export * from "openevals/types";

export type GraphTrajectory = {
  inputs?: (Record<string, unknown> | null)[];
  results: Record<string, unknown>[];
  steps: string[][];
};

export type ExtractedLangGraphThreadTrajectory = {
  inputs: (Record<string, unknown> | null)[][];
  outputs: GraphTrajectory;
};

export type TrajectoryLLMAsJudgeParams = Omit<
  Parameters<typeof createLLMAsJudge>[0],
  "prompt"
> & { prompt?: string };
