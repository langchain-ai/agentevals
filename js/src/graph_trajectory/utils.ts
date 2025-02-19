import type { StateSnapshot, Pregel } from "@langchain/langgraph/web";
import { isBaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import { _convertMessagesToOpenAIParams } from "@langchain/openai";

import type { GraphTrajectory } from "../types.js";

export const extractLangGraphTrajectoryFromSnapshots = (
  snapshots: StateSnapshot[]
) => {
  const inputs = [];
  const trajectory: GraphTrajectory = {
    results: [],
    steps: [],
  };
  let isAccumulatingSteps = false;
  for (let i = 0; i < snapshots.length; i += 1) {
    const snapshot = snapshots[i];
    const hasInterrupts = snapshot.tasks?.find((task) => {
      return task.interrupts?.length;
    });
    if (!snapshot.next?.length || hasInterrupts) {
      isAccumulatingSteps = true;
      if (hasInterrupts) {
        trajectory.results.push({});
      } else if (
        snapshot.values != null &&
        typeof snapshot.values === "object" &&
        !Array.isArray(snapshot.values) &&
        "messages" in snapshot.values &&
        Array.isArray(snapshot.values.messages)
      ) {
        const lastMessage = snapshot.values.messages.at(-1);
        if (isBaseMessage(lastMessage)) {
          // Just append the last message in the output to the results to reduce context size
          trajectory.results.push({
            messages: _convertMessagesToOpenAIParams([lastMessage]),
          });
        } else {
          trajectory.results.push({ messages: [lastMessage] });
        }
      } else {
        trajectory.results.push(snapshot.values);
      }
      trajectory.steps.push([]);
    }
    if (isAccumulatingSteps && snapshot.tasks?.length) {
      const checkpointNs = snapshot.config?.configurable?.checkpoint_ns ?? "";
      let subgraphPath = "";
      if (checkpointNs.split(":").length > 1) {
        subgraphPath = `${checkpointNs.split(":")[0]}:`;
      }
      for (const task of snapshot.tasks) {
        if (task.interrupts?.length) {
          trajectory.steps.at(-1)?.push("__interrupt__");
        }
        trajectory.steps.at(-1)?.push(`${subgraphPath}${task.name}`);
      }
    }
    if (isAccumulatingSteps) {
      if (snapshot.metadata != null && snapshot.metadata.source === "input") {
        inputs.push(snapshot.metadata.writes);
      } else if (
        i + 1 < snapshots.length &&
        snapshots[i + 1].tasks?.find((task) => task.interrupts?.length > 0)
      ) {
        inputs.push("__resuming__");
      }
    }
  }
  inputs.reverse();
  trajectory.results.reverse();
  trajectory.steps.reverse();
  for (const stepList of trajectory.steps) {
    stepList.reverse();
  }
  if (inputs.length !== trajectory.results.length) {
    console.warn(
      "Trajectory parsing may be incomplete: inputs and results have different lengths"
    );
  } else if (inputs.length !== trajectory.steps.length) {
    console.warn(
      "Trajectory parsing may be incomplete: inputs and steps have different lengths"
    );
  }
  return { inputs, outputs: trajectory };
};

export const _getLangGraphStateHistoryRecursive = async (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  graph: Pregel<any, any>,
  config: RunnableConfig
): Promise<StateSnapshot[]> => {
  const stateHistory = [];
  for await (const history of graph.getStateHistory(config)) {
    if (history.tasks?.length) {
      for (const task of history.tasks) {
        if ((task.state as RunnableConfig)?.configurable?.checkpoint_ns) {
          stateHistory.push(
            ...(await _getLangGraphStateHistoryRecursive(
              graph,
              task.state as RunnableConfig
            ))
          );
        }
      }
    }
    stateHistory.push(history);
  }
  return stateHistory;
};

export const extractLangGraphTrajectoryFromThread = async (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  graph: Pregel<any, any>,
  config: RunnableConfig
) => {
  const history = await _getLangGraphStateHistoryRecursive(graph, config);
  return extractLangGraphTrajectoryFromSnapshots(history);
};
