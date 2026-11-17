import { describe, expect, it } from "vitest";
import { createTrajectoryMatchEvaluator } from "../match.js";
import type {
  FlexibleChatCompletionMessage,
  ToolArgsMatchMode,
} from "../../types.js";

function trajectory(value: unknown): FlexibleChatCompletionMessage[] {
  return [
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          type: "function",
          id: "call-1",
          function: {
            name: "search",
            arguments: JSON.stringify({ filter: { value } }),
          },
        },
      ],
    },
  ];
}

describe.each(["exact", "subset", "superset"] as ToolArgsMatchMode[])(
  "%s tool argument matching",
  (toolArgsMatchMode) => {
    it.each([
      { actual: [], expected: {}, score: false },
      { actual: {}, expected: [], score: false },
      { actual: [1], expected: { 0: 1 }, score: false },
      { actual: [1, { a: 2 }], expected: [1, { a: 2 }], score: true },
      { actual: { a: [1] }, expected: { a: [1] }, score: true },
    ])(
      "compares nested JSON values: $actual and $expected",
      async ({ actual, expected, score }) => {
        const evaluator = createTrajectoryMatchEvaluator({ toolArgsMatchMode });
        const result = await evaluator({
          outputs: trajectory(actual),
          referenceOutputs: trajectory(expected),
        });
        expect(result.score).toBe(score);
      }
    );
  }
);

it("distinguishes arrays from objects for selected argument paths", async () => {
  const evaluator = createTrajectoryMatchEvaluator({
    toolArgsMatchOverrides: { search: ["filter.value"] },
  });
  const result = await evaluator({
    outputs: trajectory([]),
    referenceOutputs: trajectory({}),
  });
  expect(result.score).toBe(false);
});
