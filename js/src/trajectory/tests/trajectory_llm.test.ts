import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { createTrajectoryLLMAsJudge, DEFAULT_NO_REF_PROMPT } from "../llm.js";

ls.describe("Trajectory LLM", () => {
  ls.test(
    "should match trajectories",
    {
      inputs: {},
    },
    async () => {
      const evaluator = createTrajectoryLLMAsJudge({
        model: "openai:o3-mini",
      });
      const inputs = {};
      const outputs = [
        { role: "user", content: "What is the weather in SF?" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        {
          role: "assistant",
          content: "The weather in SF is 80 degrees and sunny.",
        },
      ];

      const referenceOutputs = [
        { role: "user", content: "What is the weather in SF?" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "San Francisco" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in San Francisco.",
        },
        { role: "assistant", content: "The weather in SF is 80˚ and sunny." },
      ];

      const evalResult = await evaluator({
        inputs,
        outputs,
        referenceOutputs,
      });

      expect(evalResult.key).toBe("trajectory_accuracy");
      expect(evalResult.score).toBe(true);
    }
  );

  ls.test(
    "trajectory no ref",
    { inputs: {} },
    async () => {
      const evaluator = createTrajectoryLLMAsJudge({
        prompt: DEFAULT_NO_REF_PROMPT,
        model: "openai:o3-mini",
      });
      const outputs = [
        { role: "user", content: "What is the weather in SF?" },
        {
          role: "assistant",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        { role: "assistant", content: "The weather in SF is 80 degrees and sunny." },
      ];
      const evalResult = await evaluator({
        outputs,
      });

      expect(evalResult.key).toBe("trajectory_accuracy");
      expect(evalResult.score).toBe(true);
    }
  );

  ls.test(
    "trajectory no ref bad trajectory",
    { inputs: {} },
    async () => {
      const evaluator = createTrajectoryLLMAsJudge({
        prompt: DEFAULT_NO_REF_PROMPT,
        model: "openai:o3-mini",
      });
      const outputs = [
        { role: "user", content: "What are some good restaurants in SF?" },
        {
          role: "assistant",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        { role: "assistant", content: "The weather in SF is 80 degrees and sunny." },
      ];
      const evalResult = await evaluator({
        outputs,
      });

      expect(evalResult.key).toBe("trajectory_accuracy");
      expect(evalResult.score).toBe(false);
    }
  );

  ls.test(
    "should match trajectories with inverse rubric",
    { inputs: {} },
    async () => {
      const REVERSE_PROMPT = `You are an expert data labeler.
Your task is to grade the inaccuracy of an AI agent's internal trajectory.

<Rubric>
  An inaccurate trajectory:
  - Makes no logical sense between steps
  - Shows no clear progression
  - Is not relatively efficient, though it does not need to be perfectly inefficient
  - Is not semantically equivalent to the provided reference trajectory, if present

  We are looking for bad trajectories, so score should be 0 if the trajectory contains reasonable steps for the agent to answer the input, and 1 if not.
</Rubric>

Grade the following trajectory:

<trajectory>
{outputs}
</trajectory>
{inputs}
{reference_outputs}
`;

      const evaluator = createTrajectoryLLMAsJudge({
        model: "openai:o3-mini",
        prompt: REVERSE_PROMPT,
      });
      const inputs = {};
      const outputs = [
        { role: "user", content: "What is the weather in SF?" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        {
          role: "assistant",
          content: "The weather in SF is 80 degrees and sunny.",
        },
      ];

      const referenceOutputs = [
        { role: "user", content: "What is the weather in SF?" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "San Francisco" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in San Francisco.",
        },
        { role: "assistant", content: "The weather in SF is 80˚ and sunny." },
      ];

      const evalResult = await evaluator({
        inputs,
        outputs,
        referenceOutputs,
      });

      expect(evalResult.key).toBe("trajectory_accuracy");
      expect(evalResult.score).toBe(false);
    }
  );
});
