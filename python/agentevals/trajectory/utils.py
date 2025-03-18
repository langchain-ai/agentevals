__all__ = [
    "_is_trajectory_superset",
    "_extract_tool_calls",
    "_get_matcher_for_tool_name",
]

import json

from agentevals.types import (
    ChatCompletionMessage,
    ToolArgsMatchMode,
    ToolArgsMatchOverrides,
)
from typing import Callable, Optional


def _normalize_tool_call(tool_call: dict) -> dict:
    if "function" in tool_call:
        return {
            "name": tool_call["function"]["name"],
            "args": json.loads(tool_call["function"]["arguments"]),
        }
    else:
        return tool_call


def _extract_tool_calls(messages: list[ChatCompletionMessage]) -> list[dict]:
    tool_calls: list[dict] = []
    for message in messages:
        if "tool_calls" in message:
            tool_calls.extend(
                _normalize_tool_call(tool_call) for tool_call in message["tool_calls"]
            )
    return tool_calls


def _is_trajectory_superset(
    outputs: list[ChatCompletionMessage],
    reference_outputs: list[ChatCompletionMessage],
    tool_args_match_mode: ToolArgsMatchMode,
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
):
    output_tool_calls = _extract_tool_calls(outputs)
    reference_tool_calls = _extract_tool_calls(reference_outputs)

    # Keep track of which reference tool calls have been matched
    matched_reference_calls = set()

    # For each reference tool call, find a matching output tool call
    for ref_call in reference_tool_calls:
        ref_name = ref_call["name"]
        ref_args = ref_call["args"]

        found_match = False
        for out_idx, out_call in enumerate(output_tool_calls):
            out_name = out_call["name"]

            # Names must match
            if ref_name != out_name:
                continue

            # If we're already using this output call for a different match, skip
            if out_idx in matched_reference_calls:
                continue

            # Check tool args according to match mode
            matcher = _get_matcher_for_tool_name(
                ref_name, tool_args_match_mode, tool_args_match_overrides
            )

            out_args = out_call["args"]
            if matcher(out_args, ref_args):
                matched_reference_calls.add(out_idx)
                found_match = True
                break

        # If we didn't find a match for this reference call, we're not a superset
        if not found_match:
            return False

    return True


def _exact_match(tool_call: dict, reference_tool_call: dict) -> bool:
    return tool_call == reference_tool_call


def _none_match(tool_call: dict, reference_tool_call: dict) -> bool:
    return True


def _get_matcher_for_comparison_mode(
    mode: ToolArgsMatchMode,
) -> Callable[[str, str], bool]:
    if mode == "exact":
        return _exact_match
    else:
        return _none_match


def _get_partial_matcher_on_keys(keys: list[str]) -> Callable[[str, str], bool]:
    def get_nested_value(d: dict, key_path: str):
        current = d
        for part in key_path.split("."):
            current = (
                current["function"]["arguments"]
                if part == "function.arguments"
                else current[part]
            )
        return current

    def matcher(output_call: dict, reference_call: dict) -> bool:
        return all(
            get_nested_value(output_call, key) == get_nested_value(reference_call, key)
            for key in keys
        )

    return matcher


def _get_matcher_for_tool_name(
    tool_call_name: str,
    tool_args_match_mode: ToolArgsMatchMode,
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides],
) -> Callable[[str, str], bool]:
    matcher = _get_matcher_for_comparison_mode(tool_args_match_mode)
    if tool_args_match_overrides is not None and tool_args_match_overrides.get(
        tool_call_name, None
    ):
        override = tool_args_match_overrides.get(tool_call_name)
        if isinstance(override, str):
            matcher = _get_matcher_for_comparison_mode(override)
        elif isinstance(override, Callable):
            matcher = override
        else:
            matcher = _get_partial_matcher_on_keys(override)
    return matcher
