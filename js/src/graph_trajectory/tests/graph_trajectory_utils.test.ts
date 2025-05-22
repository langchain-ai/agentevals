/* eslint-disable no-promise-executor-return */
import { expect, test } from "vitest";
import {
  Annotation,
  StateGraph,
  MemorySaver,
  Command,
  Send,
} from "@langchain/langgraph";

import { extractLangGraphTrajectoryFromThread } from "../utils.js";

test("extract trajectory", async () => {
  const checkpointer = new MemorySaver();

  const inner = new StateGraph(
    Annotation.Root({
      myKey: Annotation<string>({
        reducer: (a, b) => a + b,
        default: () => "",
      }),
      myOtherKey: Annotation<string>,
    })
  )
    .addNode("inner1", async (state) => {
      await new Promise((resolve) => setTimeout(resolve, 100));
      return { myKey: "got here", myOtherKey: state.myKey };
    })
    .addNode("inner2", (state) => ({
      myKey: " and there",
      myOtherKey: state.myKey,
    }))
    .addEdge("inner1", "inner2")
    .addEdge("__start__", "inner1")
    .compile({ interruptBefore: ["inner2"] });

  const app = new StateGraph(
    Annotation.Root({
      myKey: Annotation<string>({
        reducer: (a, b) => a + b,
        default: () => "",
      }),
    })
  )
    .addNode("inner", (state, config) => inner.invoke(state, config), {
      subgraphs: [inner],
    })
    .addNode("outer1", () => ({ myKey: " and parallel" }))
    .addNode("outer2", () => ({ myKey: " and back again" }))
    .addEdge("__start__", "inner")
    .addEdge("__start__", "outer1")
    .addEdge(["inner", "outer1"], "outer2")
    .compile({ checkpointer });

  // test invoke w/ nested interrupt
  const config = { configurable: { thread_id: "1" } };
  expect(await app.invoke({ myKey: "" }, config)).toEqual({
    myKey: " and parallel",
  });

  expect(await app.invoke(null, config)).toEqual({
    myKey: "got here and there and parallel and back again",
  });

  const trajectory = await extractLangGraphTrajectoryFromThread(app, config);
  expect(trajectory).toEqual({
    inputs: [
      {
        __start__: {
          myKey: "",
        },
      },
      {
        __start__: {
          myKey: "",
        },
      },
    ],
    outputs: {
      results: [
        {
          myKey: "got here and there",
          myOtherKey: "got here",
        },
        {
          myKey: "got here and there and parallel and back again",
        },
      ],
      steps: [
        [
          "__start__",
          "outer1",
          "inner",
          "inner:__start__",
          "inner:inner1",
          "inner:inner2",
        ],
        ["outer2"],
      ],
    },
  });
});

test("extract trajectory from graph with Command", async () => {
  const checkpointer = new MemorySaver();

  const graph = new StateGraph(
    Annotation.Root({
      items: Annotation<string[]>({
        reducer: (a, b) => a.concat(b),
        default: () => [],
      }),
      processedCount: Annotation<number>({
        reducer: (_, b) => b,
        default: () => 0,
      }),
    })
  )
    .addNode(
      "dispatcher",
      (state) => {
        // Use Command with Send to route to multiple processing nodes dynamically
        const sends = state.items.map(
          (item, index) =>
            new Send(`process_${index % 2}`, { items: [item], index })
        );
        return new Command({
          update: { processedCount: state.items.length },
          goto: sends,
        });
      },
      {
        ends: ["process_0", "process_1"],
      }
    )
    .addNode("process_0", (state) => {
      return { items: [`processed_0: ${state.items?.join(", ")}`] };
    })
    .addNode("process_1", (state) => {
      return { items: [`processed_1: ${state.items?.join(", ")}`] };
    })
    .addNode("aggregator", (state) => {
      return { items: [`final count: ${state.processedCount}`] };
    })
    .addEdge("__start__", "dispatcher")
    .addEdge(["process_0", "process_1"], "aggregator")
    .compile({ checkpointer });

  const config = { configurable: { thread_id: "3" } };

  await graph.invoke(
    {
      items: ["task1", "task2", "task3"],
    },
    config
  );

  const trajectory = await extractLangGraphTrajectoryFromThread(graph, config);

  expect(trajectory).toEqual({
    inputs: [
      {
        __start__: {
          items: ["task1", "task2", "task3"],
        },
      },
    ],
    outputs: {
      results: [
        {
          items: [
            "task1",
            "task2",
            "task3",
            "processed_0: task1",
            "processed_1: task2",
            "processed_0: task3",
            "final count: 3",
          ],
          processedCount: 3,
        },
      ],
      steps: [
        [
          "__start__",
          "dispatcher",
          "process_0",
          "process_1",
          "process_0",
          "aggregator",
        ],
      ],
    },
  });
});
