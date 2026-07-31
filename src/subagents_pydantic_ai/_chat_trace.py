"""Storage for subagent conversations that a `chat_trace_id` can resume.

A chat trace is one subagent's message history, kept so a later delegation can
continue that conversation instead of starting cold. Traces are held in memory and
bounded by an LRU, because a long-lived orchestrator would otherwise accumulate
every conversation it ever started.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

ChatTraceKey = tuple[str, str]
"""`(subagent_name, chat_trace_id)`. A trace belongs to the subagent that created
it: continuing it with a different subagent would replay one agent's history into
another."""


class ChatTraceStore:
    """LRU-bounded message histories, plus the set of traces currently running.

    A trace must finish before it can be continued. Two concurrent tasks on one
    trace would both save at the end, and the slower save would silently discard
    the faster one's history.
    """

    def __init__(self, max_traces: int) -> None:
        self.history: OrderedDict[ChatTraceKey, list[Any]] = OrderedDict()
        self._max_traces = max_traces
        self._active: set[ChatTraceKey] = set()

    def is_active(self, key: ChatTraceKey) -> bool:
        """Whether a task is currently running on this trace."""
        return key in self._active

    def mark_active(self, key: ChatTraceKey) -> None:
        """Claim the trace for a task that is about to start."""
        self._active.add(key)

    def release(self, key: ChatTraceKey) -> None:
        """Release the trace once its task has finished."""
        self._active.discard(key)

    def __contains__(self, key: ChatTraceKey) -> bool:
        return key in self.history

    def history_for(self, key: ChatTraceKey) -> list[Any] | None:
        """The stored history for a trace, refreshing its LRU recency.

        Reading counts as use, so a trace the orchestrator keeps continuing does
        not get evicted underneath it.
        """
        messages = self.history.get(key)
        if messages is not None:
            self.history.move_to_end(key)
        return messages

    def save(self, key: ChatTraceKey, messages: list[Any]) -> None:
        """Store a finished run's history, evicting the least recently used traces."""
        self.history[key] = messages
        self.history.move_to_end(key)
        while len(self.history) > self._max_traces:
            self.history.popitem(last=False)
