import { createLLMAsJudge } from "openevals/llm";

export * from "openevals/types";

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
