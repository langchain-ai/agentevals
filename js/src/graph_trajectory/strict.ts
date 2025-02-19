import { GraphTrajectory } from "../types.js";
import { _runEvaluator } from "../utils.js";

const _scorer = (params: {
  outputs: GraphTrajectory;
  referenceOutputs: GraphTrajectory;
}) => {
  const { outputs, referenceOutputs } = params;
  if (!outputs || !referenceOutputs) {
    throw new Error(
      "Strict trajectory match requires both outputs and referenceOutputs"
    );
  }
  if (outputs.steps.length !== referenceOutputs.steps.length) {
    return false;
  }
  for (let i = 0; i < outputs.steps.length; i++) {
    if (outputs.steps[i].length !== referenceOutputs.steps[i].length) {
      return false;
    }
    for (let j = 0; j < outputs.steps[i].length; j++) {
      if (outputs.steps[i][j] !== referenceOutputs.steps[i][j]) {
        return false;
      }
    }
  }
  return true;
};

/**
 * Evaluate whether an input graph trajectory strictly matches a reference graph trajectory.
 * This means that at each step, the agent took the same steps in the same order as specified in the reference trajectory.
 *
 * @param params - The parameters object
 * @param params.outputs - Actual trajectory the agent followed
 * @param params.referenceOutputs - Ideal reference trajectory the agent should have followed
 * @returns Contains a score of true if trajectory (including called tools) matches, false otherwise
 */
export const graphTrajectoryStrictMatch = ({
  outputs,
  referenceOutputs,
}: {
  outputs: GraphTrajectory;
  referenceOutputs: GraphTrajectory;
}) => {
  return _runEvaluator(
    "graph_trajectory_strict_match",
    _scorer,
    "graph_trajectory_strict_match",
    {
      outputs,
      referenceOutputs,
    }
  );
};
