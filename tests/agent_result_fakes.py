"""Shared AgentRunResult test doubles."""

from __future__ import annotations

import json
from typing import Any

_UNSET = object()


class MockUsage:
    """Mock RunUsage."""

    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
        requests: int = 1,
        details: dict[str, int] | None = None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.requests = requests
        self.details = details or {}

    def opentelemetry_attributes(self) -> dict[str, int]:
        attrs = {
            "gen_ai.usage.input_tokens": self.input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
        }
        attrs.update({f"gen_ai.usage.details.{key}": value for key, value in self.details.items()})
        return attrs


class MockResult:
    """Contract-conformant AgentRunResult test double."""

    def __init__(
        self,
        output: Any = "mock result",
        *,
        usage: Any = _UNSET,
        messages: list[Any] | None = None,
        run_id: str = "run-123",
        conversation_id: str = "conversation-123",
        traceparent: str | None = None,
    ):
        self.output = output
        self._usage = MockUsage() if usage is _UNSET else usage
        self._messages = messages or []
        self.run_id = run_id
        self.conversation_id = conversation_id
        self._traceparent_value = traceparent

    @property
    def usage(self) -> Any:
        # pydantic-ai 2.0: `AgentRunResult.usage` is a property, not a method.
        return self._usage

    def all_messages(self) -> list[Any]:
        return self._messages

    def all_messages_json(self) -> bytes:
        return json.dumps(self._messages, default=str).encode()

    def _traceparent(self, *, required: bool = True) -> str | None:
        if self._traceparent_value is None and required:
            raise AttributeError("No span was created for this agent run")
        return self._traceparent_value


class MockResultWithMessages(MockResult):
    """Mock agent result that exposes pydantic-ai-style message history."""

    def __init__(self, output: Any = "mock result", messages: list[Any] | None = None):
        super().__init__(output, messages=messages)
