from typing import Optional, TypedDict

from openevals.types import (
    ChatCompletionMessage,
    EvaluatorResult,
    FewShotExample,
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


__all__ = [
    "GraphTrajectory",
    "ChatCompletionMessage",
    "EvaluatorResult",
    "FewShotExample",
]
