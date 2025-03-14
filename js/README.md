# 🦾⚖️ AgentEvals

[Agentic applications](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/) give an LLM freedom over control flow in order to solve problems. While this freedom
can be extremely powerful, the black box nature of LLMs can make it difficult to understand how changes in one part of your agent will affect others downstream.
This makes evaluating your agents especially important.

This package contains a collection of evaluators and utilities for evaluating the performance of your agents, with a focus on **agent trajectory**, or the intermediate steps an agent takes as it runs.
It is intended to provide a good conceptual starting point for your agent's evals.

If you are looking for more general evaluation tools, please check out the companion package [`openevals`](https://github.com/langchain-ai/openevals).

## Quickstart

To get started, install `agentevals`:

```bash
npm install agentevals @langchain/core
```

This quickstart will use an evaluator powered by OpenAI's `o3-mini` model to judge your results, so you'll need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Once you've done this, you can run your first trajectory evaluator. We represent the agent's trajectory as a list of OpenAI-style messages:

```ts
import {
  createTrajectoryLLMAsJudge,
  TRAJECTORY_ACCURACY_PROMPT,
} from "agentevals";

const trajectoryEvaluator = createTrajectoryLLMAsJudge({
  prompt: TRAJECTORY_ACCURACY_PROMPT,
  model: "openai:o3-mini",
});

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

const evalResult = await trajectoryEvaluator({
  outputs,
});

console.log(evalResult);
```

```
{
    key: 'trajectory_accuracy',
    score: true,
    comment: '...'
}
```

You can see that despite the small difference in the final response and tool calls, the evaluator still returns a score of `true` since the overall trajectory is the same between the output and reference!

## Table of Contents

- [Installation](#installation)
- [Evaluators](#evaluators)
  - [Agent Trajectory](#agent-trajectory)
    - [Strict match](#strict-match)
    - [Unordered match](#unordered-match)
    - [Subset/superset match](#subset-and-superset-match)
    - [Trajectory LLM-as-judge](#trajectory-llm-as-judge)
  - [Graph Trajectory](#graph-trajectory)
    - [Graph trajectory LLM-as-judge](#graph-trajectory-llm-as-judge)
    - [Graph trajectory strict match](#graph-trajectory-strict-match)
- [Python Async Support](#python-async-support)
- [LangSmith Integration](#langsmith-integration)
  - [Pytest or Vitest/Jest](#pytest-or-vitestjest)
  - [Evaluate](#evaluate)

## Installation

You can install `agentevals` like this:

```bash
npm install agentevals @langchain/core
```

For LLM-as-judge evaluators, you will also need an LLM client. By default, `agentevals` will use [LangChain chat model integrations](https://python.langchain.com/docs/integrations/chat/) and comes with `langchain_openai` installed by default. However, if you prefer, you may use the OpenAI client directly:

```bash
npm install openai
```

It is also helpful to be familiar with some [evaluation concepts](https://docs.smith.langchain.com/evaluation/concepts) and
LangSmith's Vitest/Jest integration for running evals, which is documented [here](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest).

## Evaluators

### Agent trajectory

Agent trajectory evaluators are used to judge the trajectory of an agent's execution either against an expected trajectory or using an LLM.
These evaluators expect you to format your agent's trajectory as a list of OpenAI format dicts or as a list of LangChain `BaseMessage` classes, and handle message formatting
under the hood.

#### Strict match

The `trajectory_strict_match` evaluator, compares two trajectories and ensures that they contain the same messages
in the same order with the same tool calls. It allows for differences in message content and tool call arguments,
but requires that the selected tools at each step are the same.

```ts
import { trajectoryStrictMatch } from "agentevals";

const outputs = [
    { role: "user", content: "What is the weather in SF?" },
    {
      role: "assistant",
      tool_calls: [{
        function: { name: "get_weather", arguments: JSON.stringify({ city: "SF" }) }
      }]
    },
    { role: "tool", content: "It's 80 degrees and sunny in SF." },
    { role: "assistant", content: "The weather in SF is 80 degrees and sunny." },
];

const referenceOutputs = [
    { role: "user", content: "What is the weather in San Francisco?" },
    { role: "assistant", tool_calls: [{ function: { name: "get_weather", arguments: JSON.stringify({ city: "San Francisco" }) } }] },
    { role: "tool", content: "It's 80 degrees and sunny in San Francisco." },
];

const result = await trajectoryStrictMatch({
  outputs,
  referenceOutputs,
});

console.log(result);
```

```
{
    'key': 'trajectory_accuracy',
    'score': true,
}
```

#### Unordered match

The `trajectory_unordered_match` evaluator, compares two trajectories and ensures that they contain the same number of tool calls in any order. This is useful if you want to allow flexibility in how an agent obtains the proper information, but still do care that all information was retrieved.

```ts
import { trajectoryUnorderedMatch } from "agentevals";

const outputs = [
  { role: "user", content: "What is the weather in SF and is there anything fun happening?" },
  {
    role: "assistant",
    tool_calls: [{
      function: {
        name: "get_weather",
        arguments: JSON.stringify({ city: "SF" }),
      }
    }],
  },
  { role: "tool", content: "It's 80 degrees and sunny in SF." },
  {
    role: "assistant",
    tool_calls: [{
      function: {
        name: "get_fun_activities",
        arguments: JSON.stringify({ city: "SF" }),
      }
    }],
  },
  { role: "tool", content: "Nothing fun is happening, you should stay indoors and read!" },
  { role: "assistant", content: "The weather in SF is 80 degrees and sunny, but there is nothing fun happening." },
];

const referenceOutputs = [
  { role: "user", content: "What is the weather in SF and is there anything fun happening?" },
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_fun_activities",
          arguments: JSON.stringify({ city: "San Francisco" }),
        }
      },
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "San Francisco" }),
        }
      },
    ],
  },
  { role: "tool", content: "Nothing fun is happening, you should stay indoors and read!" },
  { role: "tool", content: "It's 80 degrees and sunny in SF." },
  { role: "assistant", content: "In SF, it's 80˚ and sunny, but there is nothing fun happening." },
];

const result = await trajectoryUnorderedMatch({
  outputs,
  referenceOutputs,
});

console.log(result)
```

```
{
    'key': 'trajectory_unordered_match',
    'score': true,
}
```

#### Subset and superset match

There are other evaluators for checking partial trajectory matches (ensuring that a trajectory contains a subset and superset of tool calls compared to a reference trajectory).

```ts
import { trajectorySubset } from "agentevals";
// import { trajectorySuperset } from "agentevals";

const outputs = [
  { role: "user", content: "What is the weather in SF and London?" },
  {
    role: "assistant",
    tool_calls: [{
      function: {
        name: "get_weather",
        arguments: JSON.stringify({ city: "SF and London" }),
      }
    }],
  },
  { role: "tool", content: "It's 80 degrees and sunny in SF, and 90 degrees and rainy in London." },
  { role: "assistant", content: "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy."},
];

const referenceOutputs = [
  { role: "user", content: "What is the weather in SF and London?" },
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "San Francisco" }),
        }
      },
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "London" }),
        }
      },
    ],
  },
  { role: "tool", content: "It's 80 degrees and sunny in San Francisco." },
  { role: "tool", content: "It's 90 degrees and rainy in London." },
  { role: "assistant", content: "The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy." },
];

const result = await trajectorySubset({
  outputs,
  referenceOutputs,
});

console.log(result)
```

```
{
    'key': 'trajectory_subset',
    'score': true,
}
```

#### Trajectory LLM-as-judge

The LLM-as-judge trajectory evaluator that uses an LLM to evaluate the trajectory. Unlike the other trajectory evaluators, it doesn't require a reference trajectory,
and supports 
This allows for more flexibility in the trajectory comparison:

```ts
import {
  createTrajectoryLLMAsJudge,
  TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
} from "agentevals";

const evaluator = createTrajectoryLLMAsJudge({
  prompt: TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
  model: "openai:o3-mini",
});

const outputs = [
  {role: "user", content: "What is the weather in SF?"},
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "SF" }),
        }
      }
    ],
  },
  {role: "tool", content: "It's 80 degrees and sunny in SF."},
  {role: "assistant", content: "The weather in SF is 80 degrees and sunny."},
]
const referenceOutputs = [
  {role: "user", content: "What is the weather in SF?"},
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "San Francisco" }),
        }
      }
    ],
  },
  {role: "tool", content: "It's 80 degrees and sunny in San Francisco."},
  {role: "assistant", content: "The weather in SF is 80˚ and sunny."},
]

const result = await evaluator({
  outputs,
  referenceOutputs,
});

console.log(result)
```

```
{
    'key': 'trajectory_accuracy',
    'score': true,
    'comment': 'The provided agent trajectory is consistent with the reference. Both trajectories start with the same user query and then correctly invoke a weather lookup through a tool call. Although the reference uses "San Francisco" while the provided trajectory uses "SF" and there is a minor formatting difference (degrees vs. ˚), these differences do not affect the correctness or essential steps of the process. Thus, the score should be: true.'
}
```

`create_trajectory_llm_as_judge` takes the same parameters as [`create_llm_as_judge`](https://github.com/langchain-ai/openevals?tab=readme-ov-file#llm-as-judge) in `openevals`, so you can customize the prompt and scoring output as needed.

In addition to `prompt` and `model`, the following parameters are also available:

- `continuous`: a boolean that sets whether the evaluator should return a float score somewhere between 0 and 1 instead of a binary score. Defaults to `False`.
- `choices`: a list of floats that sets the possible scores for the evaluator.
- `system`: a string that sets a system prompt for the judge model by adding a system message before other parts of the prompt.
- `few_shot_examples`: a list of example dicts that are appended to the end of the prompt. This is useful for providing the judge model with examples of good and bad outputs. The required structure looks like this:

```ts
const fewShotExamples = [
  {
    inputs: "What color is the sky?",
    outputs: "The sky is red.",
    reasoning: "The sky is red because it is early evening.",
    score: 1,
  }
];
```

See the [`openevals`](https://github.com/langchain-ai/openevals?tab=readme-ov-file#llm-as-judge) repo for a fully up to date list of parameters.

### Graph trajectory

For frameworks like [LangGraph](https://github.com/langchain-ai/langgraph) that model agents as graphs, it can be more convenient to represent trajectories in terms of nodes visited rather than messages. `agentevals` includes a category of evaluators called **graph trajectory** evaluators that are designed to work with this format, as well as convenient utilities for extracting trajectories from a LangGraph thread, including different conversation turns and interrupts.

The below examples will use LangGraph with the built-in formatting utility, but graph evaluators accept input in the following general format:

```ts
export type GraphTrajectory = {
  inputs?: (Record<string, unknown> | null)[];
  results: Record<string, unknown>[];
  steps: string[][];
};

const evaluator: ({ inputs, outputs, referenceOutputs, ...extra }: {
    inputs: (string | Record<string, unknown> | null)[] | {
        inputs: (string | Record<string, unknown> | null)[];
    };
    outputs: GraphTrajectory;
    referenceOutputs?: GraphTrajectory;
    [key: string]: unknown;
}) => ...
```

Where `inputs` is a list of inputs (or a dict with a key named `"inputs"`) to the graph whose items each represent the start of a new invocation in a thread, `results` representing the final output from each turn in the thread, and `steps` representing the internal steps taken for each turn.

#### Graph trajectory LLM-as-judge

This evaluator is similar to the `trajectory_llm_as_judge` evaluator, but it works with graph trajectories instead of message trajectories. Below, we set up a LangGraph agent, extract a trajectory from it using the built-in utils, and pass it to the evaluator. First, let's setup our graph, call it, and then extract the trajectory:

```ts
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver, interrupt } from "@langchain/langgraph";
import { z } from "zod";
import { extractLangGraphTrajectoryFromThread } from "agentevals";

const search = tool((_): string => {
  const userAnswer = interrupt("Tell me the answer to the question.")
  return userAnswer;
}, {
  name: "search",
  description: "Call to surf the web.",
  schema: z.object({
      query: z.string()
  })
})

const tools = [search];

// Create a checkpointer
const checkpointer = new MemorySaver();

// Create the React agent
const graph = createReactAgent({
  llm: new ChatOpenAI({ model: "gpt-4o-mini" }),
  tools,
  checkpointer,
});

// Invoke the graph with initial message
await graph.invoke(
  { messages: [{ role: "user", content: "what's the weather in sf?" }] },
  { configurable: { thread_id: "1" } }
);

// Resume the agent with a new command (simulating human-in-the-loop)
await graph.invoke(
  { messages: [{ role: "user", content: "It is rainy and 70 degrees!" }] },
  { configurable: { thread_id: "1" } }
);

const extractedTrajectory = await extractLangGraphTrajectoryFromThread(
  graph,
  { configurable: { thread_id: "1" } },
);

console.log(extractedTrajectory);
```

```
{
  'inputs': [{
      '__start__': {
          'messages': [
              {'role': 'user', 'content': "what's the weather in sf?"}
          ]}
      }, 
      '__resuming__': {
          'messages': [
              {'role': 'user', 'content': 'It is rainy and 70 degrees!'}
          ]}
      ],
      'outputs': {
          'results': [
            {},
            {
                'messages': [
                    {'role': 'ai', 'content': 'The current weather in San Francisco is rainy, with a temperature of 70 degrees.'}
                ]
            }
        ],
        'steps': [
            ['__start__', 'agent', 'tools', '__interrupt__'],
            ['agent']
        ]
    }
}
```

Now, we can pass the extracted trajectory to the evaluator:

```ts
import { createGraphTrajectoryLLMAsJudge } from "agentevals";

const graphTrajectoryEvaluator = createGraphTrajectoryLLMAsJudge({
    model: "openai:o3-mini",
})

const res = await graphTrajectoryEvaluator(
    inputs=extractedTrajectory.inputs,
    outputs=extractedTrajectory.outputs,
)

console.log(res);
```

```
{
  'key': 'graph_trajectory_accuracy',
  'score': True,
  'comment': 'The overall process follows a logical progression: the conversation begins with the user’s request, the agent then processes the request through its own internal steps (including calling tools), interrupts to obtain further input, and finally resumes to provide a natural language answer. Each step is consistent with the intended design in the rubric, and the overall path is relatively efficient and semantically aligns with a typical query resolution trajectory. Thus, the score should be: true.'
}
```

Note that though this evaluator takes the typical `inputs`, `outputs`, and `referenceOutputs` parameters, it internally combines `inputs` and `outputs` to form a `thread`. Therefore, if you want to customize the prompt, your prompt should also contain a `thread` input variable:

```ts
const CUSTOM_PROMPT = `You are an expert data labeler.
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

{reference_outputs}
`

const graphTrajectoryEvaluator = createGraphTrajectoryLLMAsJudge({
  prompt: CUSTOM_PROMPT,
  model: "openai:o3-mini",
})
res = await graphTrajectoryEvaluator(
  inputs=extractedTrajectory.inputs,
  outputs=extractedTrajectory.outputs,
)
```

In order to format them properly into the prompt, `referenceOutputs` should be passed in as a `GraphTrajectory` object like `outputs`.

Also note that like other LLM-as-judge evaluators, you can pass extra kwargs into the evaluator to format them into the prompt.

#### Graph trajectory strict match

The `graphTrajectoryStrictMatch` evaluator is a simple evaluator that checks if the steps in the provided graph trajectory match the reference trajectory exactly.

```ts
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver, interrupt } from "@langchain/langgraph";
import { z } from "zod";
import { extractLangGraphTrajectoryFromThread, graphTrajectoryStrictMatch } from "agentevals";

const search = tool((_): string => {
  const userAnswer = interrupt("Tell me the answer to the question.")
  return userAnswer;
}, {
  name: "search",
  description: "Call to surf the web.",
  schema: z.object({
      query: z.string()
  })
})

const tools = [search];

// Create a checkpointer
const checkpointer = new MemorySaver();

// Create the React agent
const graph = createReactAgent({
  llm: new ChatOpenAI({ model: "gpt-4o-mini" }),
  tools,
  checkpointer,
});

// Invoke the graph with initial message
await graph.invoke(
  { messages: [{ role: "user", content: "what's the weather in sf?" }] },
  { configurable: { thread_id: "1" } }
);

// Resume the agent with a new command (simulating human-in-the-loop)
await graph.invoke(
  { messages: [{ role: "user", content: "It is rainy and 70 degrees!" }] },
  { configurable: { thread_id: "1" } }
);

const extractedTrajectory = await extractLangGraphTrajectoryFromThread(
  graph,
  { configurable: { thread_id: "1" } },
);

const referenceTrajectory = {
  results: [],
  steps: [["__start__", "agent", "tools", "__interrupt__"], ["agent"]],
}

const result = await graphTrajectoryStrictMatch({
  outputs: trajectory.outputs,
  referenceOutputs: referenceOutputs!,
});

console.log(result);
```

```
{
  'key': 'graph_trajectory_strict_match',
  'score': True,
}
```
## LangSmith Integration

For tracking experiments over time, you can log evaluator results to [LangSmith](https://smith.langchain.com/), a platform for building production-grade LLM applications that includes tracing, evaluation, and experimentation tools.

LangSmith currently offers two ways to run evals. We'll give a quick example of how to run evals using both.

### Pytest or Vitest/Jest

First, follow [these instructions](https://docs.smith.langchain.com/evaluation/how_to_guides/vitest_jest) to set up LangSmith's Vitest/Jest runner,
setting appropriate environment variables:

```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING="true"
```


Then, set up a file named `test_trajectory.eval.ts` with the following contents:

```ts
import * as ls from "langsmith/vitest";
// import * as ls from "langsmith/jest";

import { createTrajectoryLLMAsJudge } from "agentevals";

const trajectoryEvaluator = createTrajectoryLLMAsJudge({
  model: "openai:o3-mini",
});

ls.describe("trajectory accuracy", () => {
  ls.test("accurate trajectory", {
    inputs: {
      messages: [
        {
          role: "user",
          content: "What is the weather in SF?"
        }
      ]
    },
    referenceOutputs: {
      messages: [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": JSON.stringify({"city": "San Francisco"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
        {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
      ],
    },
  }, async ({ inputs, referenceOutputs }) => {
    const outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": JSON.stringify({"city": "SF"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
        {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
    ];
    ls.logOutputs({ messages: outputs });

    await trajectoryEvaluator({
      inputs,
      outputs,
      referenceOutputs,
    });
  });
});
```

Now, run the eval with your runner of choice:

```bash
vitest run test_trajectory.eval.ts
```


Feedback from the prebuilt evaluator will be automatically logged in LangSmith as a table of results like this in your terminal:

![Terminal results](/static/img/pytest_output.png)

And you should also see the results in the experiment view in LangSmith:

![LangSmith results](/static/img/langsmith_results.png)

### Evaluate

Alternatively, you can [create a dataset in LangSmith](https://docs.smith.langchain.com/evaluation/concepts#dataset-curation) and use your created evaluators with LangSmith's [`evaluate`](https://docs.smith.langchain.com/evaluation#8-run-and-view-results) function:

```ts
import { evaluate } from "langsmith/evaluation";
import { createTrajectoryLLMAsJudge, TRAJECTORY_ACCURACY_PROMPT } from "agentevals";

const trajectoryEvaluator = createTrajectoryLLMAsJudge({
  model: "openai:o3-mini",
  prompt: TRAJECTORY_ACCURACY_PROMPT
});

await evaluate(
  (inputs) => [
        {role: "user", content: "What is the weather in SF?"},
        {
            role: "assistant",
            tool_calls: [
                {
                    function: {
                        name: "get_weather",
                        arguments: json.dumps({"city": "SF"}),
                    }
                }
            ],
        },
        {role: "tool", content: "It's 80 degrees and sunny in SF."},
        {role: "assistant", content: "The weather in SF is 80 degrees and sunny."},
    ],
  {
    data: datasetName,
    evaluators: [trajectoryEvaluator],
  }
);
```

## Thank you!

We hope that `agentevals` helps make evaluating your LLM agents easier!

If you have any questions, comments, or suggestions, please open an issue or reach out to us on X [@LangChainAI](https://x.com/langchainai).
