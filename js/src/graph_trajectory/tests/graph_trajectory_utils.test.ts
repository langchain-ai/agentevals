import { expect, test } from "vitest";
import { Annotation, StateGraph, MemorySaver } from "@langchain/langgraph";

import { extractLangGraphTrajectoryFromThread } from "../utils.js";

test("trajectory match", async () => {
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
          ":__start__",
          ":outer1",
          ":inner",
          "inner:__start__",
          "inner:inner1",
          "inner:inner2",
        ],
        [":outer2"],
      ],
    },
  });
});
