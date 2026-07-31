"""Telemetry captured from a finished subagent run onto its `TaskHandle`.

Everything here is best-effort, matching pydantic-ai's own instrumentation, which
warns on cost and serialization failures rather than propagating them. A run that
produced an answer must never be reported as `FAILED` because collecting its
metadata raised.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from subagents_pydantic_ai.types import TaskHandle

logger = logging.getLogger(__name__)

_TRACEPARENT_FIELDS = 4
"""A W3C traceparent is `version-trace_id-span_id-flags`."""


def serialize_output(output: Any) -> str:
    """Render a subagent's output as text, preserving structure where it exists.

    Pydantic models and dataclasses become JSON so the parent model receives the
    fields rather than a `repr`; everything else falls back to `str`.
    """
    if hasattr(output, "model_dump_json"):
        return str(output.model_dump_json())
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        return json.dumps(dataclasses.asdict(output), default=str)
    return str(output)


def capture_message_history(
    result: Any,
    on_message_history: Callable[[list[Any]], None] | None,
) -> None:
    """Hand a successful run's message history to the chat-trace store.

    On failure the previously saved history is kept, so continuing the trace
    resumes from the last point that was successfully saved.
    """
    if on_message_history is None:
        return

    try:
        all_messages = getattr(result, "all_messages", None)
        if all_messages is None:
            return

        messages: Any = all_messages()
        if messages:
            on_message_history(list(messages))
    except Exception as e:
        logger.warning("Failed to capture subagent message history: %s", e)


def result_traceparent(result: Any) -> str | None:
    """The run's W3C traceparent, when instrumentation produced one."""
    traceparent = getattr(result, "_traceparent", None)
    if traceparent is None:
        return None

    value = traceparent(required=False)
    return value if isinstance(value, str) and value else None


def model_responses(result: Any) -> list[Any]:
    """Every model response in the run, in order, including the final one."""
    responses: list[Any] = []
    for message in result.all_messages():
        if getattr(message, "kind", None) == "response":
            responses.append(message)

    try:
        response = result.response
    except (AttributeError, ValueError):
        response = None
    if response is not None and all(response is not item for item in responses):
        responses.append(response)
    return responses


def capture_result_observability(handle: TaskHandle, result: Any) -> None:
    """Copy a finished run's telemetry onto `handle`, letting failures propagate.

    `capture_observability` is the wrapper callers should use; this is the raw
    version, kept separate so tests can assert on collection failures.
    """
    handle.usage = result.usage
    handle.message_history = result.all_messages_json().decode()
    handle.run_id = result.run_id
    handle.conversation_id = result.conversation_id

    handle.traceparent = result_traceparent(result)
    if handle.traceparent is not None:
        parts = handle.traceparent.split("-")
        if len(parts) >= _TRACEPARENT_FIELDS:
            handle.trace_id = parts[1]
            handle.span_id = parts[2]

    responses = model_responses(result)
    if responses:
        # The final response produced the returned output, so it owns the per-run
        # model and provider metadata. Aggregates (`cost`, `tool_call_counts`) are
        # summed across every response instead, which is what a multi-model run
        # needs.
        response = responses[-1]
        handle.model_name = response.model_name
        handle.provider_name = response.provider_name
        handle.provider_url = response.provider_url
        handle.provider_response_id = response.provider_response_id
        handle.provider_details = response.provider_details
        handle.finish_reason = response.finish_reason

    total_cost = Decimal("0")
    has_cost = False
    tool_call_counts: dict[str, int] = {}
    for response in responses:
        if response.model_name:
            try:
                total_cost += response.cost().total_price
                has_cost = True
            except LookupError:
                # genai-prices has no entry for this model/provider pair.
                pass

        for tool_call in response.tool_calls:
            tool_call_counts[tool_call.tool_name] = tool_call_counts.get(tool_call.tool_name, 0) + 1

    handle.cost = total_cost if has_cost else None
    handle.tool_call_counts = tool_call_counts


def capture_observability(handle: TaskHandle, result: Any) -> None:
    """Copy a finished run's telemetry onto `handle` without ever failing the task."""
    try:
        capture_result_observability(handle, result)
    except Exception as e:
        logger.warning("Failed to capture subagent observability: %s", e)
