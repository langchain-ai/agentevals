__all__ = ["_run_evaluator", "_arun_evaluator"]

from openevals.types import EvaluatorResult
from openevals.utils import (
    _run_evaluator as _base_run_evaluator,
    _arun_evaluator as _base_arun_evaluator,
)

from typing import Any, Callable


def _run_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    return _base_run_evaluator(
        run_name=run_name,
        scorer=scorer,
        feedback_key=feedback_key,
        ls_framework="agentevals",
        **kwargs,
    )


async def _arun_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    return await _base_arun_evaluator(
        run_name=run_name,
        scorer=scorer,
        feedback_key=feedback_key,
        ls_framework="agentevals",
        **kwargs,
    )
