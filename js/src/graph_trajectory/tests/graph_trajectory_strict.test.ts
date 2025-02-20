import * as ls from "langsmith/vitest";
import { expect } from "vitest";
import {
  MemorySaver,
  Command,
  interrupt,
  Annotation,
  StateGraph,
} from "@langchain/langgraph";

import { graphTrajectoryStrictMatch } from "../strict.js";
import { extractLangGraphTrajectoryFromThread } from "../utils.js";

ls.describe(
  "graph_trajectory_strict_match",
  () => {
    ls.test(
      "should match the reference trajectory",
      {
        inputs: {},
        referenceOutputs: {
          results: [
            {},
            {
              myKey: "It is rainy and 70 degrees!",
            },
          ],
          steps: [["__start__", "agent", "interrupt", "__interrupt__"], []],
        },
      },
      async ({ referenceOutputs }) => {
        const graph = new StateGraph(
          Annotation.Root({
            myKey: Annotation<string>,
          })
        )
          .addNode("agent", async () => {
            return {
              myKey: "hello",
            };
          })
          .addNode("interrupt", async () => {
            const res = interrupt("Tell me the answer to the question.");
            return { myKey: res };
          })
          .addEdge("__start__", "agent")
          .addEdge("agent", "interrupt")
          .compile({ checkpointer: new MemorySaver() });
        const config = {
          configurable: {
            thread_id: "1",
          },
        };
        await graph.invoke(
          {
            myKey: "foo",
          },
          config
        );
        await graph.invoke(
          new Command({ resume: "It is rainy and 70 degrees!" }),
          config
        );
        const trajectory = await extractLangGraphTrajectoryFromThread(
          graph,
          config
        );
        const result = await graphTrajectoryStrictMatch({
          outputs: trajectory.outputs,
          referenceOutputs: referenceOutputs!,
        });
        expect(result.score).toBe(true);
      }
    );
  },
  {
    enableTestTracking: false,
  }
);
