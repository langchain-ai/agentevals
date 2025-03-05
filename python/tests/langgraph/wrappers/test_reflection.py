import math
import pytest

from agentevals.langgraph.wrappers.reflection import wrap_graph_with_reflection

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langsmith import testing as t
from langgraph.prebuilt import create_react_agent


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


@tool
def sin(a: float) -> float:
    """Take the sine of a number."""
    return math.sin(a)


@tool
def cos(a: float) -> float:
    """Take the cosine of a number."""
    return math.cos(a)


@tool
def radians(a: float) -> float:
    """Convert degrees to radians."""
    return math.radians(a)


@tool
def exponentiation(a: float, b: float) -> float:
    """Raise one number to the power of another."""
    return a**b


@tool
def sqrt(a: float) -> float:
    """Take the square root of a number."""
    return math.sqrt(a)


@tool
def ceil(a: float) -> float:
    """Round a number up to the nearest integer."""
    return math.ceil(a)


@pytest.mark.langsmith
@pytest.mark.skip(reason="Long running and expensive")
def test_trajectory_match():
    # Initialize agent
    llm = init_chat_model("openai:gpt-4o-mini", temperature=0.1)
    tools = [
        sin,
        cos,
        radians,
        ceil,
        exponentiation,
        sqrt,
        add,
        multiply,
        divide,
        subtract,
    ]

    agent = wrap_graph_with_reflection(graph=create_react_agent(llm, tools))

    query = (
        "A batter hits a baseball at 45.847 m/s at an angle of "
        "23.474° above the horizontal. The outfielder, who starts facing the batter, picks up the baseball as it lands, "
        "then throws it back towards the batter at 24.12 m/s at an angle of 39.12 degrees. "
        "How far is the baseball from where the batter originally hit it? "
        "Assume zero air resistance."
    )

    t.log_inputs({"messages": query})

    last_message = None
    for step in agent.stream(
        {"messages": query}, stream_mode="updates", config={"recursion_limit": 50}
    ):
        for _, update in step.items():
            for message in update.get("messages", []):
                message.pretty_print()
                last_message = message

    assert "98" in last_message.content
