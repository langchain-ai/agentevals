from agentevals.trajectory.match import create_trajectory_match_evaluator

from agentevals.types import EvaluatorResult, ChatCompletionMessage

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import json
import pytest


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_trajectory_match(feedback_key, match_mode):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80 degrees and sunny."
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80˚ and sunny."
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(
        key=feedback_key,
        score=True,
        comment=None,
        metadata=None,
    )


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_trajectory_with_different_tool_message_order(feedback_key, match_mode):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(
        key=feedback_key,
        score=True,
        comment=None,
        metadata=None,
    )


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("trajectory_unordered_match", "unordered", 1.0),
        ("trajectory_superset_match", "superset", 1.0),
        ("trajectory_subset_match", "subset", 1.0),
        ("trajectory_strict_match", "strict", 0.0),
    ],
)
def test_trajectory_with_different_message_count(feedback_key, match_mode, score):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("trajectory_unordered_match", "unordered", 0.0),
        ("trajectory_superset_match", "superset", 0.0),
        ("trajectory_subset_match", "subset", 1.0),
        ("trajectory_strict_match", "strict", 0.0),
    ],
)
def test_trajectory_subset_tool_call(feedback_key, match_mode, score):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 9000 degrees and hallucinating.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_exact_matcher_with_different_called_tools(feedback_key, match_mode):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80 degrees and sunny."
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "accuweather_forecast",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80˚ and sunny."
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=False, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("trajectory_unordered_match", "unordered", 0.0),
        ("trajectory_superset_match", "superset", 1.0),
        ("trajectory_subset_match", "subset", 0.0),
        ("trajectory_strict_match", "strict", 0.0),
    ],
)
def test_trajectory_with_extra_tool_calls_and_override(feedback_key, match_mode, score):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
        tool_args_match_overrides={"get_weather": "ignore"},
    )
    inputs = {}
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF and London"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool",
            content="It's 80 degrees and sunny in SF, and 90 degrees and rainy in London.",
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("trajectory_unordered_match", "unordered", 0.0),
        ("trajectory_superset_match", "superset", 0.0),
        ("trajectory_subset_match", "subset", 1.0),
        ("trajectory_strict_match", "strict", 0.0),
    ],
)
def test_trajectory_with_subset_tool_calls_and_override(
    feedback_key, match_mode, score
):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
        tool_args_match_overrides={"get_weather": "ignore"},
    )
    inputs = {}
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF and London"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool",
            content="It's 80 degrees and sunny in SF, and 90 degrees and rainy in London.",
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy.",
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_trajectory_match_with_langchain_messages_and_override(
    feedback_key, match_mode
):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=match_mode,
        tool_args_match_overrides={
            "get_weather": lambda x, y: (
                x["city"] == "SF" or x["city"] == "San Francisco"
            )
            and (y["city"] == "SF" or y["city"] == "San Francisco")
        },
    )
    inputs = {}
    outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1234",
                    "name": "get_weather",
                    "args": {"city": "SF"},
                }
            ],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage(content="The weather in SF is 80 degrees and sunny."),
    ]
    reference_outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="Let me check that for you!",
            tool_calls=[
                {
                    "id": "4321",
                    "name": "get_weather",
                    "args": {"city": "San Francisco"},
                }
            ],
        ),
        ToolMessage(
            tool_call_id="4321", content="It's 80 degrees and sunny in San Francisco."
        ),
        AIMessage(content="The weather in SF is 80˚ and sunny."),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(
        key=feedback_key,
        score=True,
        comment=None,
        metadata=None,
    )


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_trajectory_match_with_langchain_messages_failure(feedback_key, match_mode):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1234",
                    "name": "get_weather",
                    "args": {"city": "SF"},
                }
            ],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage(content="The weather in SF is 80 degrees and sunny."),
    ]
    reference_outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="Let me check that for you!",
            tool_calls=[
                {
                    "id": "4321",
                    "name": "accuweather_forecast",
                    "args": {"city": "San Francisco"},
                }
            ],
        ),
        ToolMessage(
            tool_call_id="4321", content="It's 80 degrees and sunny in San Francisco."
        ),
        AIMessage(content="The weather in SF is 80˚ and sunny."),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=False, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "trajectory_match_mode, score",
    [
        ("unordered", False),
        ("superset", True),
        ("subset", False),
        ("strict", False),
    ],
)
def test_trajectory_match_with_overrides(trajectory_match_mode, score):
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
                    "function": {
                        "name": "fetch_user_flight_information",
                        "arguments": "{}",
                    },
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "fetch_user_flight_information",
            "tool_call_id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
            "content": json.dumps(
                [
                    {
                        "ticket_no": "7240005432906569",
                        "book_ref": "C46E9F",
                        "flight_id": 19250,
                        "flight_no": "LX0112",
                        "departure_airport": "CDG",
                        "arrival_airport": "BSL",
                        "scheduled_departure": "2025-03-22T18:34:40Z",
                        "scheduled_arrival": "2025-03-22T20:34:40Z",
                        "seat_no": "18E",
                        "fare_conditions": "Economy",
                    }
                ]
            ),
        },
        {
            "role": "assistant",
            "content": "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
            "role": "user",
            "content": "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "f6ff5419-c03f-4543-b67d-72693c94b2ca",
                    "function": {
                        "name": "search_flights",
                        "arguments": json.dumps(
                            {
                                "start_time": "2025-03-22T18:34:40Z",
                                "departure_airport": "CDG",
                                "arrival_airport": "BSL",
                            }
                        ),
                    },
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "search_flights",
            "tool_call_id": "f6ff5419-c03f-4543-b67d-72693c94b2ca",
            "content": json.dumps(
                [
                    {
                        "flight_id": 19229,
                        "flight_no": "LX0112",
                        "scheduled_departure": "2025-03-22T19:34:40Z",
                        "scheduled_arrival": "2025-03-22T21:34:40Z",
                        "departure_airport": "CDG",
                        "arrival_airport": "BSL",
                        "status": "Scheduled",
                        "aircraft_code": "SU9",
                    },
                    {
                        "flight_id": 19232,
                        "flight_no": "LX0112",
                        "scheduled_departure": "2025-03-22T20:34:40Z",
                        "scheduled_arrival": "2025-03-22T22:34:40Z",
                        "departure_airport": "CDG",
                        "arrival_airport": "BSL",
                        "status": "Scheduled",
                        "aircraft_code": "SU9",
                    },
                ]
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "4a286aff-199a-4152-99b1-df1ca07c920e",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps({"query": "flight upgrades"}),
                    },
                },
                {
                    "type": "function",
                    "id": "00000000-0000-0000-0000-000000000000",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps({"query": "first class"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "4a286aff-199a-4152-99b1-df1ca07c920e",
            "content": "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "00000000-0000-0000-0000-000000000000",
            "content": "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
            "role": "assistant",
            "content": "The next flight after that is LX0112 from CDG to BSL is in 4 hours. However, we do not currently allow upgrades to first class. Confirming that I should book it for you anyway?",
        },
    ]

    reference_outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
                    "function": {
                        "name": "fetch_user_flight_information",
                        "arguments": "{}",
                    },
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "fetch_user_flight_information",
            "tool_call_id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
            "content": '[{"ticket_no": "7240005432906569", "book_ref": "C46E9F", "flight_id": 19250, "flight_no": "LX0112", "departure_airport": "CDG", "arrival_airport": "BSL", "scheduled_departure": "2025-03-20T15:00:00-07:00", "scheduled_arrival": "2025-03-20T16:00:00-07:00", "seat_no": "18E", "fare_conditions": "Economy"}]',
        },
        {
            "role": "assistant",
            "content": "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
            "role": "user",
            "content": "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
            "role": "assistant",
            "name": "flight_agent",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": '{"query": "upgrade to first class"}',
                    },
                },
                {
                    "type": "function",
                    "id": "00000000-0000-0000-0000-000000000000",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps({"query": "foo"}),
                    },
                },
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
            "content": "...",
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "00000000-0000-0000-0000-000000000000",
            "content": "...",
        },
        {
            "role": "assistant",
            "name": "flight_agent",
            "content": "Ok, it looks like upgrades to first class are possible. What date would you like to change your flight to?",
        },
    ]

    evaluator_no_overrides = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
    )

    evaluator_no_overrides_result = evaluator_no_overrides(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert not evaluator_no_overrides_result["score"]

    def lookup_policy_query_matcher(tool_args: dict, reference_tool_args: dict):
        if reference_tool_args.get(
            "query", {}
        ) and "upgrade" in reference_tool_args.get("query", {}):
            return "upgrade" in tool_args.get("query", {})
        # Ignore for other policy query matches
        return True

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_overrides={
            "lookup_policy": lookup_policy_query_matcher,
        },
    )
    evaluator_result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert evaluator_result["score"] == score


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "trajectory_match_mode",
    [
        ("unordered"),
        ("superset"),
        ("subset"),
        ("strict"),
    ],
)
def test_trajectory_match_with_nested_field_overrides(trajectory_match_mode):
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
                    "function": {
                        "name": "fetch_user_flight_information",
                        "arguments": json.dumps({"user_id": "123"}),
                    },
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "fetch_user_flight_information",
            "tool_call_id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
            "content": json.dumps(
                [
                    {
                        "ticket_no": "7240005432906569",
                        "book_ref": "C46E9F",
                        "flight_id": 19250,
                        "flight_no": "LX0112",
                        "departure_airport": "CDG",
                        "arrival_airport": "BSL",
                        "scheduled_departure": "2025-03-22T18:34:40Z",
                        "scheduled_arrival": "2025-03-22T20:34:40Z",
                        "seat_no": "18E",
                        "fare_conditions": "Economy",
                    }
                ]
            ),
        },
        {
            "role": "assistant",
            "content": "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
            "role": "user",
            "content": "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "4a286aff-199a-4152-99b1-df1ca07c920e",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps(
                            {
                                "query": "flight upgrades",
                                "time": {
                                    "start": "2025-03-22T18:34:40Z",
                                    "end": "2025-03-22T20:34:40Z",
                                },
                            }
                        ),
                    },
                },
                {
                    "type": "function",
                    "id": "00000000-0000-0000-0000-000000000000",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps(
                            {
                                "query": "first class",
                                "time": {
                                    "start": "2025-03-22T18:34:40Z",
                                    "end": "2025-03-22T20:34:40Z",
                                },
                            }
                        ),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "4a286aff-199a-4152-99b1-df1ca07c920e",
            "content": "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "00000000-0000-0000-0000-000000000000",
            "content": "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
            "role": "assistant",
            "content": "The next flight after that is LX0112 from CDG to BSL is in 4 hours. However, we do not currently allow upgrades to first class. Confirming that I should book it for you anyway?",
        },
    ]

    reference_outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
                    "function": {
                        "name": "fetch_user_flight_information",
                        "arguments": json.dumps({"user_id": "123"}),
                    },
                },
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "fetch_user_flight_information",
            "tool_call_id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
            "content": '[{"ticket_no": "7240005432906569", "book_ref": "C46E9F", "flight_id": 19250, "flight_no": "LX0112", "departure_airport": "CDG", "arrival_airport": "BSL", "scheduled_departure": "2025-03-20T15:00:00-07:00", "scheduled_arrival": "2025-03-20T16:00:00-07:00", "seat_no": "18E", "fare_conditions": "Economy"}]',
        },
        {
            "role": "assistant",
            "content": "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
            "role": "user",
            "content": "Update it to the next flight after that and bump me to first class if there is availability.",
        },
        {
            "role": "assistant",
            "name": "flight_agent",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": '{"query": "foo", "time": {"start": "2025-03-22T18:34:40Z", "end": "baz"}}',
                    },
                },
                {
                    "type": "function",
                    "id": "00000000-0000-0000-0000-000000000000",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps(
                            {
                                "query": "bar",
                                "time": {"start": "2025-03-22T18:34:40Z", "end": "baz"},
                            }
                        ),
                    },
                },
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "cb2f81d3-382a-46ce-8fa0-a7ece7a75de1",
            "content": "...",
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "00000000-0000-0000-0000-000000000000",
            "content": "...",
        },
        {
            "role": "assistant",
            "name": "flight_agent",
            "content": "Ok, it looks like upgrades to first class are possible. What date would you like to change your flight to?",
        },
    ]

    evaluator_no_overrides = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
    )

    evaluator_no_overrides_result = evaluator_no_overrides(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert not evaluator_no_overrides_result["score"]

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        # Only match on time.start for lookup_policy
        tool_args_match_overrides={
            "lookup_policy": ["time.start"],
        },
    )
    evaluator_result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert evaluator_result["score"]


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "tool_args_match_mode, score",
    [
        ("exact", False),
        ("ignore", True),
        ("subset", False),
        ("superset", True),
    ],
)
def test_tool_args_match_mode_superset(tool_args_match_mode, score):
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "123",
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps(
                            {"is_cool": True, "flight_no": "LX0112"}
                        ),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    reference_outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "321",
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps({"flight_no": "LX0112"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    evaluator = create_trajectory_match_evaluator(
        tool_args_match_mode=tool_args_match_mode,
    )
    evaluator_result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert evaluator_result["score"] == score


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "tool_args_match_mode, score",
    [
        ("exact", False),
        ("ignore", True),
        ("subset", True),
        ("superset", False),
    ],
)
def test_tool_args_match_mode_subset(tool_args_match_mode, score):
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "123",
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps({"flight_no": "LX0112"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    reference_outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "321",
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps({"flight_no": "LX0112", "foo": "bar"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    evaluator = create_trajectory_match_evaluator(
        tool_args_match_mode=tool_args_match_mode,
    )
    evaluator_result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert evaluator_result["score"] == score


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "tool_args_match_mode, score",
    [
        ("exact", True),
        ("ignore", True),
        ("subset", True),
        ("superset", True),
    ],
)
def test_tool_args_match_mode_exact(tool_args_match_mode, score):
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "123",
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps({"flight_no": "LX0112"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    reference_outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "321",
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps({"flight_no": "LX0112"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    evaluator = create_trajectory_match_evaluator(
        tool_args_match_mode=tool_args_match_mode,
    )
    evaluator_result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert evaluator_result["score"] == score
