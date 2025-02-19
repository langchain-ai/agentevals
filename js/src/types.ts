export * from "openevals/types";

export type GraphTrajectory = {
  inputs?: Record<string, unknown>[];
  results: Record<string, unknown>[];
  steps: string[][];
};

export type ExtractedLangGraphThreadTrajectory = {
  inputs: unknown[];
  outputs: GraphTrajectory;
};
