/* eslint-disable @typescript-eslint/no-explicit-any */
import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { createTrajectoryMatchEvaluator } from "../match.js";

ls.describe("trajectory", () => {
  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
    },
  ])("trajectory exact match", async ({ trajectoryMatchMode, feedbackKey }) => {
    const outputs = [
      {
        role: "user",
        content: "What is the weather in SF?",
      },
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
        content: "It's 80 degrees and sunny in SF.",
      },
      {
        role: "assistant",
        content: "The weather in SF is 80 degrees and sunny.",
      },
    ];
    const referenceOutputs = [
      {
        role: "user",
        content: "What is the weather in SF?",
      },
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
        content: "It's 80˚ and sunny in San Francisco.",
      },
      {
        role: "assistant",
        content: "The weather in San Francisco is 80˚ and sunny.",
      },
    ];
    const evaluator = createTrajectoryMatchEvaluator({
      trajectoryMatchMode,
    });
    const result = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(result).toBeDefined();
    expect(result.key).toBe(feedbackKey);
    expect(result.score).toBe(true);
    expect(result.comment).toBeUndefined();
  });

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
    },
  ])(
    "different tool message order",
    async ({ trajectoryMatchMode, feedbackKey }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
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
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in SF.",
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        },
      ];
      const referenceOutputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in SF.",
        },
        {
          role: "assistant",
          content:
            "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        },
      ];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(true);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
      score: true,
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
      score: true,
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
      score: true,
    },
  ])(
    "different message count",
    async ({ trajectoryMatchMode, feedbackKey, score }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
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
        {
          role: "tool",
          content: "It's 80 degrees and sunny in SF.",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        },
      ];
      const referenceOutputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in SF.",
        },
        {
          role: "assistant",
          content:
            "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        },
      ];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
      score: true,
    },
  ])(
    "trajectory subset tool call",
    async ({ trajectoryMatchMode, feedbackKey, score }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
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
        {
          role: "tool",
          content: "It's 80 degrees and sunny in SF.",
        },
        {
          role: "assistant",
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 9000 degrees and hallucinating.",
        },
      ];
      const referenceOutputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        },
      ];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
      score: false,
    },
  ])(
    "different called tools",
    async ({ trajectoryMatchMode, feedbackKey, score }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF?",
        },
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
        {
          role: "tool",
          content: "It's 80 degrees and sunny in SF.",
        },
        {
          role: "assistant",
          content: "The weather in SF is 80 degrees and sunny.",
        },
      ];
      const referenceOutputs = [
        {
          role: "user",
          content: "What is the weather in SF?",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "accuweather_forecast",
                arguments: JSON.stringify({ city: "San Francisco" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in San Francisco.",
        },
        {
          role: "assistant",
          content: "The weather in SF is 80˚ and sunny.",
        },
      ];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
      score: true,
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
      score: false,
    },
  ])(
    "trajectory with extra tool calls",
    async ({ trajectoryMatchMode, feedbackKey, score }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
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
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in San Francisco.",
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy.",
        },
      ];
      const referenceOutputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
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
          content:
            "It's 80 degrees and sunny in San Francisco, and 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        },
      ];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
    },
  ])(
    "trajectory match with langchain messages",
    async ({ trajectoryMatchMode, feedbackKey }) => {
      const outputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "",
          tool_calls: [
            {
              id: "1234",
              name: "get_weather",
              args: { city: "San Francisco" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "1234",
          content: "It's 80 degrees and sunny in SF.",
        }),
        new AIMessage("The weather in SF is 80 degrees and sunny."),
      ];
      const referenceOutputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "Let me check that for you!",
          tool_calls: [
            {
              id: "4321",
              name: "get_weather",
              args: { city: "San Francisco" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "4321",
          content: "It's 80 degrees and sunny in San Francisco.",
        }),
        new AIMessage("The weather in SF is 80˚ and sunny."),
      ];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(true);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
    },
  ])(
    "trajectory match with langchain messages failure",
    async ({ trajectoryMatchMode, feedbackKey }) => {
      const outputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "",
          tool_calls: [
            {
              id: "1234",
              name: "get_weather",
              args: { city: "SF" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "1234",
          content: "It's 80 degrees and sunny in SF.",
        }),
        new AIMessage("The weather in SF is 80 degrees and sunny."),
      ];
      const referenceOutputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "Let me check that for you!",
          tool_calls: [
            {
              id: "4321",
              name: "accuweather_forecast",
              args: { city: "San Francisco" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "4321",
          content: "It's 80 degrees and sunny in San Francisco.",
        }),
        new AIMessage("The weather in SF is 80˚ and sunny."),
      ];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(false);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      toolArgsMatchMode: "exact",
      score: false,
    },
    {
      inputs: {},
      toolArgsMatchMode: "ignore",
      score: true,
    },
  ])("trajectory match strict params", async ({ toolArgsMatchMode, score }) => {
    const outputs = [
      new HumanMessage("What is the weather in SF?"),
      new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "1234",
            name: "get_weather",
            args: { city: "SF" },
          },
        ],
      }),
      new ToolMessage({
        content: "It's 80 degrees and sunny in SF.",
        tool_call_id: "1234",
      }),
      new AIMessage("The weather in SF is 80 degrees and sunny."),
    ];

    const referenceOutputs = [
      new HumanMessage("What is the weather in SF?"),
      new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "1234",
            name: "get_weather",
            args: { city: "San Francisco" },
          },
        ],
      }),
      new ToolMessage({
        content: "It's 80 degrees and sunny in SF.",
        tool_call_id: "1234",
      }),
      new AIMessage("The weather in SF is 80 degrees and sunny."),
    ];

    const evaluator = createTrajectoryMatchEvaluator({
      trajectoryMatchMode: "strict",
      toolArgsMatchMode,
    });

    const result = await evaluator({
      outputs,
      referenceOutputs,
    });

    expect(result).toEqual({
      key: "trajectory_strict_match",
      score,
      comment: undefined,
    });
  });
  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "unordered" as const,
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset" as const,
      feedbackKey: "trajectory_superset_match",
      score: true,
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset" as const,
      feedbackKey: "trajectory_subset_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "strict" as const,
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
  ])(
    "trajectory match with overrides",
    async ({ trajectoryMatchMode, feedbackKey, score }) => {
      const outputs = [
        { role: "user", content: "Hi there, what time is my flight?" },
        {
          role: "assistant",
          tool_calls: [
            {
              type: "function",
              id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
              function: {
                name: "fetch_user_flight_information",
                arguments: "{}",
              },
            },
          ],
          content: "",
        },
        {
          role: "tool",
          name: "fetch_user_flight_information",
          tool_call_id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
          content: JSON.stringify([
            {
              ticket_no: "7240005432906569",
              book_ref: "C46E9F",
              flight_id: 19250,
              flight_no: "LX0112",
              departure_airport: "CDG",
              arrival_airport: "BSL",
              scheduled_departure: "2025-03-22T18:34:40Z",
              scheduled_arrival: "2025-03-22T20:34:40Z",
              seat_no: "18E",
              fare_conditions: "Economy",
            },
          ]),
        },
        {
          role: "assistant",
          content:
            "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
          role: "user",
          content:
            "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
          role: "assistant",
          tool_calls: [
            {
              type: "function",
              id: "f6ff5419-c03f-4543-b67d-72693c94b2ca",
              function: {
                name: "search_flights",
                arguments: JSON.stringify({
                  start_time: "2025-03-22T18:34:40Z",
                  departure_airport: "CDG",
                  arrival_airport: "BSL",
                }),
              },
            },
          ],
          content: "",
        },
        {
          role: "tool",
          name: "search_flights",
          tool_call_id: "f6ff5419-c03f-4543-b67d-72693c94b2ca",
          content: JSON.stringify([
            {
              flight_id: 19229,
              flight_no: "LX0112",
              scheduled_departure: "2025-03-22T19:34:40Z",
              scheduled_arrival: "2025-03-22T21:34:40Z",
              departure_airport: "CDG",
              arrival_airport: "BSL",
              status: "Scheduled",
              aircraft_code: "SU9",
            },
            {
              flight_id: 19232,
              flight_no: "LX0112",
              scheduled_departure: "2025-03-22T20:34:40Z",
              scheduled_arrival: "2025-03-22T22:34:40Z",
              departure_airport: "CDG",
              arrival_airport: "BSL",
              status: "Scheduled",
              aircraft_code: "SU9",
            },
          ]),
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              type: "function",
              id: "4a286aff-199a-4152-99b1-df1ca07c920e",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({ query: "flight upgrades" }),
              },
            },
            {
              type: "function",
              id: "00000000-0000-0000-0000-000000000000",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({ query: "first class" }),
              },
            },
          ],
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "4a286aff-199a-4152-99b1-df1ca07c920e",
          content:
            "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "00000000-0000-0000-0000-000000000000",
          content:
            "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
          role: "assistant",
          content:
            "The next flight after that is LX0112 from CDG to BSL is in 4 hours. However, we do not currently allow upgrades to first class. Confirming that I should book it for you anyway?",
        },
      ];

      const referenceOutputs = [
        { role: "user", content: "Hi there, what time is my flight?" },
        {
          role: "assistant",
          tool_calls: [
            {
              type: "function",
              id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
              function: {
                name: "fetch_user_flight_information",
                arguments: "{}",
              },
            },
          ],
          content: "",
        },
        {
          role: "tool",
          name: "fetch_user_flight_information",
          tool_call_id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
          content:
            '[{"ticket_no": "7240005432906569", "book_ref": "C46E9F", "flight_id": 19250, "flight_no": "LX0112", "departure_airport": "CDG", "arrival_airport": "BSL", "scheduled_departure": "2025-03-20T15:00:00-07:00", "scheduled_arrival": "2025-03-20T16:00:00-07:00", "seat_no": "18E", "fare_conditions": "Economy"}]',
        },
        {
          role: "assistant",
          content:
            "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
          role: "user",
          content:
            "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
          role: "assistant",
          name: "flight_agent",
          tool_calls: [
            {
              type: "function",
              id: "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
              function: {
                name: "lookup_policy",
                arguments: '{"query": "upgrade to first class"}',
              },
            },
            {
              type: "function",
              id: "00000000-0000-0000-0000-000000000000",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({ query: "foo" }),
              },
            },
          ],
          content: "",
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
          content: "...",
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "00000000-0000-0000-0000-000000000000",
          content: "...",
        },
        {
          role: "assistant",
          name: "flight_agent",
          content:
            "Ok, it looks like upgrades to first class are possible. What date would you like to change your flight to?",
        },
      ];
      const evaluatorNoOverrides = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });

      const evaluatorNoOverridesResult = await evaluatorNoOverrides({
        outputs,
        referenceOutputs,
      });
      expect(evaluatorNoOverridesResult.score).toBe(false);

      const lookupPolicyQueryMatcher = (
        toolArgs: Record<string, any>,
        referenceToolArgs: Record<string, any>
      ) => {
        if (
          referenceToolArgs.query &&
          referenceToolArgs.query.includes("upgrade")
        ) {
          return toolArgs.query?.includes("upgrade") ?? false;
        }
        return true;
      };

      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchOverrides: {
          lookup_policy: lookupPolicyQueryMatcher,
        },
      });

      const evaluatorResult = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(evaluatorResult.score).toBe(score);
      expect(evaluatorResult.key).toBe(feedbackKey);
    }
  );

  ls.test.each([
    { trajectoryMatchMode: "unordered", inputs: {} },
    { trajectoryMatchMode: "superset", inputs: {} },
    { trajectoryMatchMode: "subset", inputs: {} },
    { trajectoryMatchMode: "strict", inputs: {} },
  ])(
    "trajectory match with nested field overrides",
    async ({ trajectoryMatchMode }) => {
      const outputs = [
        { role: "user", content: "Hi there, what time is my flight?" },
        {
          role: "assistant",
          tool_calls: [
            {
              type: "function",
              id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
              function: {
                name: "fetch_user_flight_information",
                arguments: JSON.stringify({ user_id: "123" }),
              },
            },
          ],
          content: "",
        },
        {
          role: "tool",
          name: "fetch_user_flight_information",
          tool_call_id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
          content: JSON.stringify([
            {
              ticket_no: "7240005432906569",
              book_ref: "C46E9F",
              flight_id: 19250,
              flight_no: "LX0112",
              departure_airport: "CDG",
              arrival_airport: "BSL",
              scheduled_departure: "2025-03-22T18:34:40Z",
              scheduled_arrival: "2025-03-22T20:34:40Z",
              seat_no: "18E",
              fare_conditions: "Economy",
            },
          ]),
        },
        {
          role: "assistant",
          content:
            "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
          role: "user",
          content:
            "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              type: "function",
              id: "4a286aff-199a-4152-99b1-df1ca07c920e",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({
                  query: "flight upgrades",
                  time: {
                    start: "2025-03-22T18:34:40Z",
                    end: "2025-03-22T20:34:40Z",
                  },
                }),
              },
            },
            {
              type: "function",
              id: "00000000-0000-0000-0000-000000000000",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({
                  query: "first class",
                  time: {
                    start: "2025-03-22T18:34:40Z",
                    end: "2025-03-22T20:34:40Z",
                  },
                }),
              },
            },
          ],
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "4a286aff-199a-4152-99b1-df1ca07c920e",
          content:
            "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "00000000-0000-0000-0000-000000000000",
          content:
            "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
          role: "assistant",
          content:
            "The next flight after that is LX0112 from CDG to BSL is in 4 hours. However, we do not currently allow upgrades to first class. Confirming that I should book it for you anyway?",
        },
      ];
      const referenceOutputs = [
        { role: "user", content: "Hi there, what time is my flight?" },
        {
          role: "assistant",
          tool_calls: [
            {
              type: "function",
              id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
              function: {
                name: "fetch_user_flight_information",
                arguments: JSON.stringify({ user_id: "123" }),
              },
            },
          ],
          content: "",
        },
        {
          role: "tool",
          name: "fetch_user_flight_information",
          tool_call_id: "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
          content:
            '[{"ticket_no": "7240005432906569", "book_ref": "C46E9F", "flight_id": 19250, "flight_no": "LX0112", "departure_airport": "CDG", "arrival_airport": "BSL", "scheduled_departure": "2025-03-20T15:00:00-07:00", "scheduled_arrival": "2025-03-20T16:00:00-07:00", "seat_no": "18E", "fare_conditions": "Economy"}]',
        },
        {
          role: "assistant",
          content:
            "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
          role: "user",
          content:
            "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
          role: "assistant",
          name: "flight_agent",
          tool_calls: [
            {
              type: "function",
              id: "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
              function: {
                name: "lookup_policy",
                arguments:
                  '{"query": "foo", "time": {"start": "2025-03-22T18:34:40Z", "end": "baz"}}',
              },
            },
            {
              type: "function",
              id: "00000000-0000-0000-0000-000000000000",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({
                  query: "bar",
                  time: { start: "2025-03-22T18:34:40Z", end: "baz" },
                }),
              },
            },
          ],
          content: "",
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
          content: "...",
        },
        {
          role: "tool",
          name: "lookup_policy",
          tool_call_id: "00000000-0000-0000-0000-000000000000",
          content: "...",
        },
        {
          role: "assistant",
          name: "flight_agent",
          content:
            "Ok, it looks like upgrades to first class are possible. What date would you like to change your flight to?",
        },
      ];

      const evaluatorNoOverrides = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
      });

      const evaluatorNoOverridesResult = await evaluatorNoOverrides({
        outputs,
        referenceOutputs,
      });
      expect(evaluatorNoOverridesResult.score).toBe(false);

      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchOverrides: {
          lookup_policy: ["time.start"],
        },
      });

      const evaluatorResult = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(evaluatorResult.score).toBe(true);
    }
  );

  ls.test.each([
    { inputs: {}, toolArgsMatchMode: "exact", score: false },
    { inputs: {}, toolArgsMatchMode: "ignore", score: true },
    { inputs: {}, toolArgsMatchMode: "subset", score: false },
    { inputs: {}, toolArgsMatchMode: "superset", score: true },
  ])("tool_args_match_mode superset", async ({ toolArgsMatchMode, score }) => {
    const outputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "123",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ is_cool: true, flight_no: "LX0112" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ];
    const referenceOutputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "321",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ flight_no: "LX0112" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ];
    const evaluator = createTrajectoryMatchEvaluator({
      toolArgsMatchMode: toolArgsMatchMode as any,
    });
    const evaluatorResult = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(evaluatorResult.score).toBe(score);
  });

  ls.test.each([
    { inputs: {}, toolArgsMatchMode: "exact", score: false },
    { inputs: {}, toolArgsMatchMode: "ignore", score: true },
    { inputs: {}, toolArgsMatchMode: "subset", score: true },
    { inputs: {}, toolArgsMatchMode: "superset", score: false },
  ])("tool_args_match_mode subset", async ({ toolArgsMatchMode, score }) => {
    const outputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "123",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ flight_no: "LX0112" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ];
    const referenceOutputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "321",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ flight_no: "LX0112", foo: "bar" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ];
    const evaluator = createTrajectoryMatchEvaluator({
      toolArgsMatchMode: toolArgsMatchMode as any,
    });
    const evaluatorResult = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(evaluatorResult.score).toBe(score);
  });

  ls.test.each([
    { inputs: {}, toolArgsMatchMode: "exact", score: true },
    { inputs: {}, toolArgsMatchMode: "ignore", score: true },
    { inputs: {}, toolArgsMatchMode: "subset", score: true },
    { inputs: {}, toolArgsMatchMode: "superset", score: true },
  ])("tool_args_match_mode exact", async ({ toolArgsMatchMode, score }) => {
    const outputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "123",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ flight_no: "LX0112" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ];
    const referenceOutputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "321",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ flight_no: "LX0112" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ];
    const evaluator = createTrajectoryMatchEvaluator({
      toolArgsMatchMode: toolArgsMatchMode as any,
    });
    const evaluatorResult = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(evaluatorResult.score).toBe(score);
  });
});
