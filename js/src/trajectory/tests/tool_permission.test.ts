import { test, expect } from "vitest";

import { createTrajectoryToolPermissionEvaluator } from "../toolPermission.js";

function assistantWithTools(...names: string[]) {
  return {
    role: "assistant",
    content: "",
    tool_calls: names.map((name) => ({
      function: { name, arguments: "{}" },
    })),
  };
}

test("all calls authorized", async () => {
  const evaluator = createTrajectoryToolPermissionEvaluator({
    allowedTools: ["search_kb", "reply"],
  });
  const result = await evaluator({
    outputs: [assistantWithTools("search_kb", "reply")] as never,
  });
  expect(result.score).toBe(1);
});

test("unauthorized tool", async () => {
  const evaluator = createTrajectoryToolPermissionEvaluator({
    allowedTools: ["search_kb"],
  });
  const result = await evaluator({
    outputs: [assistantWithTools("search_kb", "delete_account")] as never,
  });
  expect(result.score).toBe(0.5);
});

test("denied tool takes precedence over allow", async () => {
  const evaluator = createTrajectoryToolPermissionEvaluator({
    allowedTools: ["search_kb", "wire_transfer"],
    deniedTools: ["wire_transfer"],
  });
  const result = await evaluator({
    outputs: [assistantWithTools("wire_transfer")] as never,
  });
  expect(result.score).toBe(0);
});

test("no tools called passes", async () => {
  const evaluator = createTrajectoryToolPermissionEvaluator({
    allowedTools: ["search_kb"],
  });
  const result = await evaluator({
    outputs: [{ role: "assistant", content: "Here is your answer." }] as never,
  });
  expect(result.score).toBe(1);
});

test("denylist only", async () => {
  const evaluator = createTrajectoryToolPermissionEvaluator({
    deniedTools: ["rm_rf"],
  });
  const result = await evaluator({
    outputs: [assistantWithTools("safe_tool", "rm_rf")] as never,
  });
  expect(result.score).toBe(0.5);
});

test("requires a policy", () => {
  expect(() => createTrajectoryToolPermissionEvaluator({})).toThrow();
});
