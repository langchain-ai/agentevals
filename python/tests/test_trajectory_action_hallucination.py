from agentevals.trajectory.action_hallucination import (
    create_action_hallucination_evaluator,
    create_async_action_hallucination_evaluator,
)

from agentevals.types import EvaluatorResult, ChatCompletionMessage

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pathlib import Path
import json
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "action_hallucination"
TOOLS = ["send_email", "lookup_order", "issue_refund"]
CASES = json.loads((FIXTURES / "cases.json").read_text())


def _refund_trajectory(email_status):
    return [
        HumanMessage(content="Refund order 42 and email me a confirmation."),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "issue_refund", "args": {"order_id": "42"}},
                {"id": "call_2", "name": "send_email", "args": {"to": "me"}},
            ],
        ),
        ToolMessage(content="Refund issued.", tool_call_id="call_1"),
        ToolMessage(
            content="SMTP gateway timeout"
            if email_status == "error"
            else "Email sent.",
            tool_call_id="call_2",
            status=email_status,
        ),
        AIMessage(content="I issued the refund and I have emailed you a confirmation."),
    ]


@pytest.mark.langsmith
def test_action_hallucination_failed_tool_with_langchain_messages():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    result = evaluator(outputs=_refund_trajectory("error"))
    assert result == EvaluatorResult(
        key="action_hallucination",
        score=False,
        comment='send_email (tool_failed): "I have emailed you a confirmation"',
        metadata=None,
    )


@pytest.mark.langsmith
def test_action_hallucination_clean_with_langchain_messages():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    result = evaluator(outputs=_refund_trajectory("success"))
    assert result == EvaluatorResult(
        key="action_hallucination",
        score=True,
        comment=None,
        metadata=None,
    )


@pytest.mark.langsmith
def test_action_hallucination_not_called():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    outputs = [
        HumanMessage(content="Where is my order?"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "lookup_order", "args": {"order_id": "42"}}
            ],
        ),
        ToolMessage(content="Order 42 shipped.", tool_call_id="call_1"),
        AIMessage(content="Your order has shipped. I've sent you the tracking email."),
    ]
    result = evaluator(outputs={"messages": outputs})
    assert result["score"] is False
    assert result["comment"] == (
        'send_email (not_called): "I\'ve sent you the tracking email"'
    )


@pytest.mark.langsmith
def test_action_hallucination_unknown_tool_without_tools_param():
    # Without `tools`, send_email is unknown because it never appears in the trajectory.
    evaluator = create_action_hallucination_evaluator()
    outputs = [
        ChatCompletionMessage(role="user", content="Email me the receipt."),
        ChatCompletionMessage(
            role="assistant", content="I have emailed you the receipt."
        ),
    ]
    assert evaluator(outputs=outputs)["score"] is True


@pytest.mark.langsmith
def test_action_hallucination_openai_messages_uncalled_tool():
    evaluator = create_action_hallucination_evaluator(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email to the customer",
                },
            },
            {"name": "lookup_order", "description": "Look up an order"},
        ]
    )
    outputs = [
        ChatCompletionMessage(role="user", content="Email me the receipt."),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "lookup_order",
                        "arguments": json.dumps({"order_id": "42"}),
                    },
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="Order 42: $30.", tool_call_id="call_1"
        ),
        ChatCompletionMessage(
            role="assistant", content="I have emailed you the receipt."
        ),
    ]
    assert evaluator(outputs=outputs) == EvaluatorResult(
        key="action_hallucination",
        score=False,
        comment='send_email (not_called): "I have emailed you the receipt"',
        metadata=None,
    )


def _openai_email_trajectory(tool_message):
    return [
        ChatCompletionMessage(role="user", content="Email me the receipt."),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "send_email",
                        "arguments": json.dumps({"to": "me"}),
                    },
                }
            ],
        ),
        tool_message,
        ChatCompletionMessage(
            role="assistant", content="I have emailed you the receipt."
        ),
    ]


@pytest.mark.langsmith
def test_action_hallucination_openai_messages_without_status_are_successful():
    # OpenAI-format tool messages carry no status, so a failure described only in the
    # content is not detected.
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    outputs = _openai_email_trajectory(
        ChatCompletionMessage(
            role="tool", content="Error: SMTP gateway timeout", tool_call_id="call_1"
        )
    )
    assert evaluator(outputs=outputs)["score"] is True


@pytest.mark.langsmith
def test_action_hallucination_openai_messages_respect_error_status():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    outputs = _openai_email_trajectory(
        {
            "role": "tool",
            "content": "SMTP gateway timeout",
            "tool_call_id": "call_1",
            "status": "error",
        }
    )
    assert evaluator(outputs=outputs)["score"] is False


@pytest.mark.langsmith
def test_action_hallucination_no_claims():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    outputs = [
        ChatCompletionMessage(role="user", content="Can you refund me?"),
        ChatCompletionMessage(
            role="assistant", content="I can refund you once you confirm the order."
        ),
    ]
    assert evaluator(outputs=outputs)["score"] is True


@pytest.mark.langsmith
def test_action_hallucination_empty_trajectory():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    assert evaluator(outputs=[])["score"] is True


@pytest.mark.langsmith
def test_action_hallucination_single_message_dict():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    result = evaluator(
        outputs={"role": "assistant", "content": "I have emailed you the receipt."}
    )
    assert (
        result["comment"] == 'send_email (not_called): "I have emailed you the receipt"'
    )


@pytest.mark.langsmith
def test_action_hallucination_content_blocks():
    evaluator = create_action_hallucination_evaluator(tools=TOOLS)
    outputs = [
        HumanMessage(content="Email me the receipt."),
        AIMessage(
            content=[
                {"type": "text", "text": "Done."},
                {"type": "text", "text": "I have emailed you the receipt."},
            ]
        ),
    ]
    assert evaluator(outputs=outputs)["score"] is False


def test_action_hallucination_invalid_tool():
    with pytest.raises(ValueError):
        create_action_hallucination_evaluator(tools=[{"description": "No name"}])


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_action_hallucination_failed_tool():
    evaluator = create_async_action_hallucination_evaluator(tools=TOOLS)
    result = await evaluator(outputs=_refund_trajectory("error"))
    assert result == EvaluatorResult(
        key="action_hallucination",
        score=False,
        comment='send_email (tool_failed): "I have emailed you a confirmation"',
        metadata=None,
    )


def _case_trajectory(case):
    calls = [
        {"id": f"call_{i}", "name": entry["tool"], "args": {}}
        for i, entry in enumerate(case["ledger"])
    ]
    results = [
        ToolMessage(
            content="",
            tool_call_id=f"call_{i}",
            status="success" if entry["status"] == "ok" else "error",
        )
        for i, entry in enumerate(case["ledger"])
    ]
    return [
        HumanMessage(content="Help me with my order."),
        *([AIMessage(content="", tool_calls=calls)] if calls else []),
        *results,
        AIMessage(content=case["text"]),
    ]


@pytest.mark.parametrize("case", [pytest.param(c, id=c["name"]) for c in CASES])
def test_action_hallucination_cases(case):
    evaluator = create_action_hallucination_evaluator(tools=case["tools"])
    result = evaluator(outputs=_case_trajectory(case))
    expected = case["expected"]
    assert result["score"] is (not expected)
    assert result["comment"] == (
        ", ".join(f'{e["tool"]} ({e["reason"]}): "{e["text"]}"' for e in expected)
        or None
    )
