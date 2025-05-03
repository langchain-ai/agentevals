# 🦾⚖️ AgentEvals

[Agentic applications](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/) give an LLM freedom over control flow in order to solve problems. While this freedom
can be extremely powerful, the black box nature of LLMs can make it difficult to understand how changes in one part of your agent will affect others downstream.
This makes evaluating your agents especially important.

This package contains a collection of evaluators and utilities for evaluating the performance of your agents, with a focus on **agent trajectory**, or the intermediate steps an agent takes as it runs.
It is intended to provide a good conceptual starting point for your agent's evals.

If you are looking for more general evaluation tools, please check out the companion package [`openevals`](https://github.com/langchain-ai/openevals).

# Quickstart

To get started, install `agentevals`:

```bash
pip install agentevals
```

This quickstart will use an evaluator powered by OpenAI's `o3-mini` model to judge your results, so you'll need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Once you've done this, you can run your first trajectory evaluator. We represent the agent's trajectory as a list of OpenAI-style messages:

```python
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

trajectory_evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    model="openai:o3-mini",
)

# This is a fake trajectory, in reality you would run your agent to get a real trajectory
outputs = [
    {"role": "user", "content": "What is the weather in SF?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "SF"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
]

eval_result = trajectory_evaluator(
  outputs=outputs,
)

print(eval_result)
```

```
{
  'key': 'trajectory_accuracy',
  'reasoning': 'The trajectory accurately follows the user's request for weather information in SF. Initially, the assistant recognizes the goal (providing weather details), then it efficiently makes a tool call to get the weather, and finally it communicates the result clearly. All steps demonstrate logical progression and efficiency. Thus, the score should be: true.',
  'score': true
}
```

You can see that the evaluator returns a score of `true` since the overall trajectory is a reasonable path for the agent to take to answer the user's question.

For more details on this evaluator, including how to customize it, see the section on [trajectory LLM-as-judge](#trajectory-llm-as-judge).

# Table of Contents

- [Installation](#installation)
- [Evaluators](#evaluators)
  - [Agent Trajectory Match](#agent-trajectory-match)
    - [Strict match](#strict-match)
    - [Unordered match](#unordered-match)
    - [Subset/superset match](#subset-and-superset-match)
    - [Tool args match modes](#tool-args-match-modes)
  - [Trajectory LLM-as-judge](#trajectory-llm-as-judge)
  - [Graph Trajectory](#graph-trajectory)
    - [Graph trajectory LLM-as-judge](#graph-trajectory-llm-as-judge)
    - [Graph trajectory strict match](#graph-trajectory-strict-match)
- [Python Async Support](#python-async-support)
- [LangSmith Integration](#langsmith-integration)
  - [Pytest or Vitest/Jest](#pytest-or-vitestjest)
  - [Evaluate](#evaluate)

# Installation

You can install `agentevals` like this:

```bash
pip install agentevals
```

For LLM-as-judge evaluators, you will also need an LLM client. By default, `agentevals` will use [LangChain chat model integrations](https://python.langchain.com/docs/integrations/chat/) and comes with `langchain_openai` installed by default. However, if you prefer, you may use the OpenAI client directly:

```bash
pip install openai
```

It is also helpful to be familiar with some [evaluation concepts](https://docs.smith.langchain.com/evaluation/concepts) and
LangSmith's pytest integration for running evals, which is documented [here](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest).

# Evaluators

## Agent trajectory match

Agent trajectory match evaluators are used to judge the trajectory of an agent's execution either against an expected trajectory or using an LLM.
These evaluators expect you to format your agent's trajectory as a list of OpenAI format dicts or as a list of LangChain `BaseMessage` classes, and handle message formatting
under the hood.

AgentEvals offers the `create_trajectory_match_evaluator`/`createTrajectoryMatchEvaluator` and `create_async_trajectory_match_evaluator` methods for this task. You can customize their behavior in a few ways:

- Setting `trajectory_match_mode`/`trajectoryMatchMode` to [`strict`](#strict-match), [`unordered`](#unordered-match), [`subset`](#subset-and-superset-match), or [`superset`](#subset-and-superset-match) to provide the general strategy the evaluator will use to compare trajectories
- Setting [`tool_args_match_mode`](#tool-args-match-modes) and/or [`tool_args_match_overrides`](#tool-args-match-modes) to customize how the evaluator considers equality between tool calls in the actual trajectory vs. the reference. By default, only tool calls with the same arguments to the same tool are considered equal.

### Strict match

The `"strict"` `trajectory_match_mode` compares two trajectories and ensures that they contain the same messages
in the same order with the same tool calls. Note that it does allow for differences in message content:

```python
import json
from agentevals.trajectory.match import create_trajectory_match_evaluator

outputs = [
    {"role": "user", "content": "What is the weather in SF?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"}),
                }
            },
            {
                "function": {
                    "name": "accuweather_forecast",
                    "arguments": json.dumps({"city": "San Francisco"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
]
reference_outputs = [
    {"role": "user", "content": "What is the weather in San Francisco?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
    {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
]

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict"
)

result = evaluator(
    outputs=outputs, reference_outputs=reference_outputs
)

print(result)
```

```
{
    'key': 'trajectory_strict_match',
    'score': False,
    'comment': None,
}
```

`"strict"` is useful is if you want to ensure that tools are always called in the same order for a given query (e.g. a company policy lookup tool before a tool that requests vacation time for an employee).

**Note:** If you would like to configure the way this evaluator checks for tool call equality, see [this section](#tool-args-match-modes).

### Unordered match

The `"unordered"` `trajectory_match_mode` compares two trajectories and ensures that they contain the same tool calls in any order. This is useful if you want to allow flexibility in how an agent obtains the proper information, but still do care that all information was retrieved.

```python
import json
from agentevals.trajectory.match import create_trajectory_match_evaluator

inputs = {}
outputs = [
    {"role": "user", "content": "What is the weather in SF and is there anything fun happening?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "San Francisco"}),
            }
        }],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "get_fun_activities",
                "arguments": json.dumps({"city": "San Francisco"}),
            }
        }],
    },
    {"role": "tool", "content": "Nothing fun is happening, you should stay indoors and read!"},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny, but there is nothing fun happening."},
]
reference_outputs = [
    {"role": "user", "content": "What is the weather in SF and is there anything fun happening?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_fun_activities",
                    "arguments": json.dumps({"city": "San Francisco"}),
                }
            },
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"}),
                }
            },
        ],
    },
    {"role": "tool", "content": "Nothing fun is happening, you should stay indoors and read!"},
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {"role": "assistant", "content": "In SF, it's 80˚ and sunny, but there is nothing fun happening."},
]

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="unordered"
)

result = evaluator(
    outputs=outputs, reference_outputs=reference_outputs
)

print(result)
```

```
{
    'key': 'trajectory_unordered_match',
    'score': True,
    'comment': None,
}
```

`"unordered"` is useful is if you want to ensure that specific tools are called at some point in the trajectory, but you don't necessarily need them to be in message order (e.g. the agent called a company policy retrieval tool at an arbitrary point in an interaction before authorizing spend for a pizza party).

**Note:** If you would like to configure the way this evaluator checks for tool call equality, see [this section](#tool-args-match-modes).

### Subset and superset match

The `"subset"` and `"superset"` modes match partial trajectories (ensuring that a trajectory contains a subset/superset of tool calls contained in a reference trajectory).

```python
import json
from agentevals.trajectory.match import create_trajectory_match_evaluator

outputs = [
    {"role": "user", "content": "What is the weather in SF and London?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "SF and London"}),
            },
        }, {
            "function": {
                "name": "accuweather_forecast",
                "arguments": json.dumps({"city": "SF and London"}),
            }
        }],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF, and 90 degrees and rainy in London."},
    {"role": "tool", "content": "Unknown."},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy."},
]
reference_outputs = [
    {"role": "user", "content": "What is the weather in SF and London?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "SF and London"}),
                }
            },
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco, and 90 degrees and rainy in London."},
    {"role": "assistant", "content": "The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy."},
]

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="superset", # or "subset"
)

result = evaluator(
    outputs=outputs, reference_outputs=reference_outputs
)

print(result)
```

```
{
    'key': 'trajectory_superset_match',
    'score': True,
    'comment': None,
}
```

`"superset"` is useful if you want to ensure that some key tools were called at some point in the trajectory, but an agent calling extra tools is still acceptable. `"subset"` is the inverse and is useful if you want to ensure that the agent did not call any tools beyond the expected ones.

**Note:** If you would like to configure the way this evaluator checks for tool call equality, see [this section](#tool-args-match-modes).

### Tool args match modes

When checking equality between tool calls, the above evaluators will require that all tool call arguments are the exact same by default. You can configure this behavior in the following ways:

- Treating any two tool calls for the same tool as equivalent by setting `tool_args_match_mode="ignore"` (Python) or `toolArgsMatchMode: "ignore"` (TypeScript)
- Treating a tool call as equivalent if it contain as subset/superset of args compared to a reference tool call of the same name with `tool_args_match_mode="subset"/"superset"` (Python) or `toolArgsMatchMode: "subset"/"superset` (TypeScript)
- Setting custom matchers for all calls of a given tool using the `tool_args_match_overrides` (Python) or `toolArgsMatchOverrides` (TypeScript) param

You can set both of these parameters at the same time. `tool_args_match_overrides` will take precendence over `tool_args_match_mode`.

`tool_args_match_overrides`/`toolArgsMatchOverrides` takes a dictionary whose keys are tool names and whose values are either `"exact"`, `"ignore"`, a list of fields within the tool call that must match exactly, or a comparator function that takes two arguments and returns whether they are equal:

```python
ToolArgsMatchMode = Literal["exact", "ignore", "subset", "superset"]

ToolArgsMatchOverrides = dict[str, Union[ToolArgsMatchMode, list[str],  Callable[[dict, dict], bool]]]
```

Here's an example that allows case insensitivity for the arguments to a tool named `get_weather`:

```python
import json
from agentevals.trajectory.match import create_trajectory_match_evaluator

outputs = [
    {"role": "user", "content": "What is the weather in SF?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "san francisco"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
]
reference_outputs = [
    {"role": "user", "content": "What is the weather in San Francisco?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
    {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
]

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict",
    tool_args_match_mode="exact",  # Default value
    tool_args_match_overrides={
        "get_weather": lambda x, y: x["city"].lower() == y["city"].lower()
    }
)

result = evaluator(
    outputs=outputs, reference_outputs=reference_outputs
)

print(result)
```

```
{
    'key': 'trajectory_strict_match',
    'score': True,
    'comment': None,
}
```

This flexibility allows you to handle cases where you want looser equality for LLM generated arguments (`"san francisco"` to equal `"San Francisco"`) for only specific tool calls.

## Trajectory LLM-as-judge

The LLM-as-judge trajectory evaluator that uses an LLM to evaluate the trajectory. Unlike the trajectory match evaluators, it doesn't require a reference trajectory. Here's an example:

```python
import json
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

evaluator = create_trajectory_llm_as_judge(
  prompt=TRAJECTORY_ACCURACY_PROMPT,
  model="openai:o3-mini"
)
outputs = [
    {"role": "user", "content": "What is the weather in SF?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "SF"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
]
eval_result = evaluator(
    outputs=outputs,
)

print(eval_result)
```

```
{
    'key': 'trajectory_accuracy',
    'score': True,
    'comment': 'The provided agent trajectory is reasonable...'
}
```

If you have a reference trajectory, you can add an extra variable to your prompt and pass in the reference trajectory. Below, we use the prebuilt  `TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE` prompt, which contains a `reference_outputs` variable:

```python
import json
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE

evaluator = create_trajectory_llm_as_judge(
  prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
  model="openai:o3-mini"
)
outputs = [
    {"role": "user", "content": "What is the weather in SF?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "SF"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
    {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
]
reference_outputs = [
    {"role": "user", "content": "What is the weather in SF?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"}),
                }
            }
        ],
    },
    {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
    {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
]
eval_result = evaluator(
    outputs=outputs,
    reference_outputs=reference_outputs,
)

print(eval_result)
```

```
{
    'key': 'trajectory_accuracy',
    'score': True,
    'comment': 'The provided agent trajectory is consistent with the reference. Both trajectories start with the same user query and then correctly invoke a weather lookup through a tool call. Although the reference uses "San Francisco" while the provided trajectory uses "SF" and there is a minor formatting difference (degrees vs. ˚), these differences do not affect the correctness or essential steps of the process. Thus, the score should be: true.'
}
```

`create_trajectory_llm_as_judge` takes the same parameters as [`create_llm_as_judge`](https://github.com/langchain-ai/openevals?tab=readme-ov-file#llm-as-judge) in `openevals`, so you can customize the prompt and scoring output as needed.

In addition to `prompt` and `model`, the following parameters are also available:

- `continuous`: a boolean that sets whether the evaluator should return a float score somewhere between 0 and 1 instead of a binary score. Defaults to `False`.
- `choices`: a list of floats that sets the possible scores for the evaluator.
- `system`: a string that sets a system prompt for the judge model by adding a system message before other parts of the prompt.
- `few_shot_examples`: a list of example dicts that are appended to the end of the prompt. This is useful for providing the judge model with examples of good and bad outputs. The required structure looks like this:

```python
few_shot_examples = [
    {
        "inputs": "What color is the sky?",
        "outputs": "The sky is red.",
        "reasoning": "The sky is red because it is early evening.",
        "score": 1,
    }
]
```

See the [`openevals`](https://github.com/langchain-ai/openevals?tab=readme-ov-file#llm-as-judge) repo for a fully up to date list of parameters.

## Graph trajectory

For frameworks like [LangGraph](https://github.com/langchain-ai/langgraph) that model agents as graphs, it can be more convenient to represent trajectories in terms of nodes visited rather than messages. `agentevals` includes a category of evaluators called **graph trajectory** evaluators that are designed to work with this format, as well as convenient utilities for extracting trajectories from a LangGraph thread, including different conversation turns and interrupts.

The below examples will use LangGraph with the built-in formatting utility, but graph evaluators accept input in the following general format:

```python
class GraphTrajectory(TypedDict):
    # Only set when specifying reference_outputs
    inputs: Optional[list[dict]]
    results: list[dict]
    steps: list[list[str]]
    
def evaluator(
    *,
    inputs: Optional[Union[dict, list]] = None,
    outputs: GraphTrajectory,
    reference_outputs: Optional[GraphTrajectory] = None,
) -> ...
```

Where `inputs` is a list of inputs (or a dict with a key named `"inputs"`) to the graph whose items each represent the start of a new invocation in a thread, `results` representing the final output from each turn in the thread, and `steps` representing the internal steps taken for each turn.

### Graph trajectory LLM-as-judge

This evaluator is similar to the `trajectory_llm_as_judge` evaluator, but it works with graph trajectories instead of message trajectories. Below, we set up a LangGraph agent, extract a trajectory from it using the built-in utils, and pass it to the evaluator. First, let's setup our graph, call it, and then extract the trajectory:

```python
from agentevals.graph_trajectory.utils import (
    extract_langgraph_trajectory_from_thread,
)
from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from langchain_core.tools import tool

@tool
def search(query: str):
    """Call to surf the web."""
    user_answer = interrupt("Tell me the answer to the question.")
    return user_answer

tools = [search]

checkpointer = MemorySaver()
graph = create_react_agent(
    model="gpt-4o-mini",
    checkpointer=checkpointer,
    tools=[search],
)

graph.invoke(
    {"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    config={"configurable": {"thread_id": "1"}},
)
# Resume the agent with a new command, simulating a human-in-the-loop workflow
graph.invoke(
    Command(resume="It is rainy and 70 degrees!"),
    config={"configurable": {"thread_id": "1"}},
)

# Extract the trajectory from the first two thread runs
extracted_trajectory = extract_langgraph_trajectory_from_thread(
    graph, {"configurable": {"thread_id": "1"}}
)

print(extracted_trajectory)
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

```python
graph_trajectory_evaluator = create_graph_trajectory_llm_as_judge(
    model="openai:o3-mini",
)

res = graph_trajectory_evaluator(
    inputs=extracted_trajectory["inputs"],
    outputs=extracted_trajectory["outputs"],
)

print(res)
```

```
{
  'key': 'graph_trajectory_accuracy',
  'score': True,
  'comment': 'The overall process follows a logical progression: the conversation begins with the user’s request, the agent then processes the request through its own internal steps (including calling tools), interrupts to obtain further input, and finally resumes to provide a natural language answer. Each step is consistent with the intended design in the rubric, and the overall path is relatively efficient and semantically aligns with a typical query resolution trajectory. Thus, the score should be: true.'
}
```

Note that though this evaluator takes the typical `inputs`, `outputs`, and `reference_outputs` parameters, it internally combines `inputs` and `outputs` to form a `thread`. Therefore, if you want to customize the prompt, your prompt should also contain a `thread` input variable:

```python
CUSTOM_PROMPT = """You are an expert data labeler.
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
"""

evaluator = create_graph_trajectory_llm_as_judge(
    prompt=CUSTOM_PROMPT,
    model="openai:o3-mini",
)
res = await evaluator(
    inputs=extracted_trajectory["inputs"],
    outputs=extracted_trajectory["outputs"],   
)
```

In order to format them properly into the prompt, `reference_outputs` should be passed in as a `GraphTrajectory` object like `outputs`.

Also note that like other LLM-as-judge evaluators, you can pass extra params into the evaluator to format them into the prompt.

### Graph trajectory strict match

The `graph_trajectory_strict_match` evaluator is a simple evaluator that checks if the steps in the provided graph trajectory match the reference trajectory exactly.

```python
from agentevals.graph_trajectory.utils import (
    extract_langgraph_trajectory_from_thread,
)
from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from langchain_core.tools import tool

@tool
def search(query: str):
    """Call to surf the web."""
    user_answer = interrupt("Tell me the answer to the question.")
    return user_answer

tools = [search]

checkpointer = MemorySaver()
graph = create_react_agent(
    model="gpt-4o-mini",
    checkpointer=checkpointer,
    tools=[search],
)

graph.invoke(
    {"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    config={"configurable": {"thread_id": "1"}},
)
# Resume the agent with a new command, simulating a human-in-the-loop workflow
graph.invoke(
    Command(resume="It is rainy and 70 degrees!"),
    config={"configurable": {"thread_id": "1"}},
)

# Extract the trajectory from the first two thread runs
extracted_trajectory = extract_langgraph_trajectory_from_thread(
    graph, {"configurable": {"thread_id": "1"}}
)

reference_trajectory = {
    # not used for strict match
    "results": [],
    "steps": [["__start__", "agent", "tools", "__interrupt__"], ["agent"]],
}

res = graph_trajectory_strict_match(
    outputs=extracted_trajectory["outputs"],
    reference_outputs=reference_trajectory,
)

print(res)
```

```
{
  'key': 'graph_trajectory_strict_match',
  'score': True,
}
```

# Python Async Support

All `agentevals` evaluators support Python [asyncio](https://docs.python.org/3/library/asyncio.html). As a convention, evaluators that use a factory function will have `async` put immediately after `create_` in the function name (for example, `create_async_trajectory_llm_as_judge`), and evaluators used directly will end in `async` (e.g. `trajectory_strict_match_async`).

Here's an example of how to use the `create_async_llm_as_judge` evaluator asynchronously:

```python
from agentevals.trajectory.llm import create_async_trajectory_llm_as_judge

evaluator = create_async_llm_as_judge(
    prompt="What is the weather in {inputs}?",
)

result = await evaluator(inputs="San Francisco")
```

If you are using the OpenAI client directly, remember to pass in `AsyncOpenAI` as the `judge` parameter:

```python
from openai import AsyncOpenAI

evaluator = create_async_llm_as_judge(
    prompt="What is the weather in {inputs}?",
    judge=AsyncOpenAI(),
    model="o3-mini",
)

result = await evaluator(inputs="San Francisco")
```

# LangSmith Integration

For tracking experiments over time, you can log evaluator results to [LangSmith](https://smith.langchain.com/), a platform for building production-grade LLM applications that includes tracing, evaluation, and experimentation tools.

LangSmith currently offers two ways to run evals: a [pytest](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest) (Python) or [Vitest/Jest](https://docs.smith.langchain.com/evaluation/how_to_guides/vitest_jest) integration and the `evaluate` function. We'll give a quick example of how to run evals using both.

## Pytest or Vitest/Jest

First, follow [these instructions](https://docs.smith.langchain.com/evaluation/how_to_guides/pytest) to set up LangSmith's pytest runner, or these to set up [Vitest or Jest](https://docs.smith.langchain.com/evaluation/how_to_guides/vitest_jest),
setting appropriate environment variables:

```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING="true"
```

Then, set up a file named `test_trajectory.py` with the following contents:

```python
import pytest
import json

from langsmith import testing as t

from agentevals.trajectory.llm import create_trajectory_llm_as_judge

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
)

@pytest.mark.langsmith
def test_trajectory_accuracy():
    outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
        {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
    ]
    reference_outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
        {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
    ]

    t.log_inputs({})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    trajectory_evaluator(
      outputs=outputs,
      reference_outputs=reference_outputs
    )
```

Note that when creating the evaluator, we've added a `feedback_key` parameter. This will be used to name the feedback in LangSmith.

Now, run the eval with pytest:

```bash
pytest test_trajectory.py --langsmith-output
```

Feedback from the prebuilt evaluator will be automatically logged in LangSmith as a table of results like this in your terminal:

![Terminal results](/static/img/pytest_output.png)

And you should also see the results in the experiment view in LangSmith:

![LangSmith results](/static/img/langsmith_results.png)

## Evaluate

Alternatively, you can [create a dataset in LangSmith](https://docs.smith.langchain.com/evaluation/concepts#dataset-curation) and use your created evaluators with LangSmith's [`evaluate`](https://docs.smith.langchain.com/evaluation#8-run-and-view-results) function:

```python
from langsmith import Client
from agentevals.trajectory.llm import create_trajectory_llm_as_judge

client = Client()

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
)

experiment_results = client.evaluate(
    # This is a dummy target function, replace with your actual LLM-based system
    lambda inputs: "What color is the sky?",
    data="Sample dataset",
    evaluators=[
        trajectory_evaluator
    ]
)
```

# Thank you!

We hope that `agentevals` helps make evaluating your LLM agents easier!

If you have any questions, comments, or suggestions, please open an issue or reach out to us on X [@LangChainAI](https://x.com/langchainai).
