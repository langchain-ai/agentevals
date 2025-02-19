import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";

import { createGraphTrajectoryLLMAsJudge } from "../llm.js";
import { extractLangGraphTrajectoryFromThread } from "../utils.js";

const search = tool(
  async () => {
    return "It's 80 degrees and sunny in San Francisco.";
  },
  {
    name: "search",
    description: "Call to surf the web.",
    schema: z.object({
      query: z.string(),
    }),
  }
);

const tools = [search];

ls.describe("graph_trajectory_llm", () => {
  ls.test(
    "sensible_trajectory",
    {
      inputs: {},
      referenceOutputs: {},
    },
    async () => {
      const checkpointer = new MemorySaver();
      const graph = createReactAgent({
        llm: new ChatOpenAI({ model: "gpt-4o-mini" }),
        checkpointer,
        tools,
      });
      const config = { configurable: { thread_id: "1" } };
      await graph.invoke(
        { messages: [{ role: "user", content: "what's the weather in sf?" }] },
        config
      );
      const trajectory = await extractLangGraphTrajectoryFromThread(
        graph,
        config
      );
      const evaluator = createGraphTrajectoryLLMAsJudge({
        model: "openai:o3-mini",
      });
      const res = await evaluator({
        inputs: trajectory.inputs,
        outputs: trajectory.outputs,
      });
      expect(res.key).toBe("graph_trajectory_accuracy");
      expect(res.score).toBe(true);
    }
  );

  ls.test(
    "unsensible_trajectory",
    {
      inputs: {},
      referenceOutputs: {},
    },
    async () => {
      const checkpointer = new MemorySaver();
      const askjeeves = tool(
        async () => {
          return "foo";
        },
        {
          name: "askjeeves",
          description: "Call to surf the web.",
          schema: z.object({ query: z.string() }),
        }
      );
      const graph = createReactAgent({
        llm: new ChatOpenAI({ model: "gpt-4o-mini" }),
        checkpointer,
        prompt:
          "You are an evil assistant who is inefficient and calls more tools than necessary.",
        tools: [askjeeves],
      });
      const config = { configurable: { thread_id: "1" } };
      await graph.invoke(
        { messages: [{ role: "user", content: "what's the weather in sf?" }] },
        config
      );
      const trajectory = await extractLangGraphTrajectoryFromThread(
        graph,
        config
      );
      const evaluator = createGraphTrajectoryLLMAsJudge({
        model: "openai:o3-mini",
        prompt: `You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal steps in resolving a user queries.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is perfectly efficient, with no more than one tool call
  - Is semantically equivalent to the provided reference trajectory, if present
</Rubric>

<Instructions>
  Grade the following thread, evaluating whether the agent's overall steps are logical and relatively efficient.
  For the trajectory, "__start__" denotes an initial entrypoint to the agent, and "__interrupt__" corresponds to the agent
  interrupting to await additional data from another source ("human-in-the-loop"):
</Instructions>

<thread>
{thread}
</thread>

{reference_outputs}`,
      });
      const res = await evaluator({
        inputs: trajectory.inputs,
        outputs: trajectory.outputs,
      });
      expect(res.key).toBe("graph_trajectory_accuracy");
      expect(res.score).toBe(false);
    }
  );
});
