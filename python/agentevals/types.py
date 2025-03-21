from typing import Callable, Literal, Optional, Union
from typing_extensions import TypedDict

from openevals.types import (
    ChatCompletionMessage,
    EvaluatorResult,
    FewShotExample,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
)


# Trajectory extracted from agent
class GraphTrajectory(TypedDict):
    inputs: Optional[list[dict]]
    results: list[dict]
    steps: list[list[str]]


# Trajectory extracted from a LangGraph thread
class ExtractedLangGraphThreadTrajectory(TypedDict):
    inputs: list
    outputs: GraphTrajectory


ToolArgsMatchMode = Literal["exact", "ignore"]

ToolArgsMatchOverrides = dict[
    str, Union[ToolArgsMatchMode, list[str], Callable[[dict, dict], bool]]
]

__all__ = [
    "GraphTrajectory",
    "ChatCompletionMessage",
    "EvaluatorResult",
    "SimpleEvaluator",
    "SimpleAsyncEvaluator",
    "FewShotExample",
    "ToolArgsMatchMode",
    "ToolArgsMatchOverrides",
]
