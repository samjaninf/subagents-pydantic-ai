"""Main subagent toolset with dual-mode execution support.

This module provides the core toolset for delegating tasks to subagents.
It supports both synchronous (blocking) and asynchronous (background)
execution modes, with automatic mode selection based on task characteristics.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset

from subagents_pydantic_ai.message_bus import InMemoryMessageBus, TaskManager
from subagents_pydantic_ai.prompts import (
    ANSWER_SUBAGENT_DESCRIPTION,
    CHECK_TASK_DESCRIPTION,
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    HARD_CANCEL_TASK_DESCRIPTION,
    LIST_ACTIVE_TASKS_DESCRIPTION,
    SEND_MESSAGE_TO_SUBAGENT_DESCRIPTION,
    SOFT_CANCEL_TASK_DESCRIPTION,
    SUBAGENT_SYSTEM_PROMPT,
    TASK_TOOL_DESCRIPTION,
    WAIT_TASKS_DESCRIPTION,
    get_task_instructions_prompt,
)
from subagents_pydantic_ai.protocols import SubAgentDepsProtocol
from subagents_pydantic_ai.retry import RetryConfig, run_with_retry
from subagents_pydantic_ai.types import (
    AgentMessage,
    AskUserCallback,
    CompiledSubAgent,
    ExecutionMode,
    MessageType,
    SubAgentConfig,
    TaskCharacteristics,
    TaskHandle,
    TaskPriority,
    TaskStatus,
    ToolsetFactory,
    UsageLimitsFactory,
    decide_execution_mode,
)

logger = logging.getLogger(__name__)


def _serialize_output(output: Any) -> str:
    """Serialize subagent output preserving structure for Pydantic models.

    For Pydantic models (BaseModel), returns JSON via `model_dump_json()`.
    For dataclasses with `__dataclass_fields__`, returns JSON via `json.dumps`.
    For everything else, returns `str(output)`.
    """
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json()  # type: ignore[no-any-return]
    if hasattr(output, "__dataclass_fields__"):
        import dataclasses
        import json

        return json.dumps(dataclasses.asdict(output), default=str)
    return str(output)


def _capture_message_history(
    result: Any,
    on_message_history: Callable[[list[Any]], None] | None,
) -> None:
    """Capture a successful run's message history for a chat trace.

    Best-effort, like `_capture_observability_best_effort`: a raising
    `all_messages()` (or save callback) must never flip a successful run to
    `FAILED`. On failure the previous saved history (if any) is kept, so a
    continued trace resumes from the last successfully saved point.
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


def _get_result_traceparent(result: Any) -> str | None:
    traceparent = getattr(result, "_traceparent", None)
    if traceparent is None:
        return None

    value = traceparent(required=False)
    return value if isinstance(value, str) and value else None


def _iter_model_responses(result: Any) -> list[Any]:
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


def _capture_result_observability(handle: TaskHandle, result: Any) -> None:
    """Copy result observability onto a task handle."""
    handle.usage = result.usage
    handle.message_history = result.all_messages_json().decode()
    handle.run_id = result.run_id
    handle.conversation_id = result.conversation_id

    handle.traceparent = _get_result_traceparent(result)
    if handle.traceparent is not None:
        parts = handle.traceparent.split("-")
        if len(parts) >= 4:
            handle.trace_id = parts[1]
            handle.span_id = parts[2]

    responses = _iter_model_responses(result)
    if responses:
        # Final response wins for per-run model/provider metadata: it is the
        # model response that produced the returned output. In a multi-model
        # run, earlier responses' model/provider metadata is not captured on
        # the handle; aggregate fields such as `cost` and `tool_call_counts`
        # are summed across all responses.
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
            except (AssertionError, LookupError):
                pass

        for tool_call in response.tool_calls:
            tool_call_counts[tool_call.tool_name] = tool_call_counts.get(tool_call.tool_name, 0) + 1

    handle.cost = total_cost if has_cost else None
    handle.tool_call_counts = tool_call_counts


def _capture_observability_best_effort(handle: TaskHandle, result: Any) -> None:
    """Capture telemetry without ever failing the task.

    Observability is best-effort (matching pydantic-ai's own instrumentation,
    which warns on cost/serialization failures rather than propagating). A run
    that succeeded must not be reported as ``FAILED`` just because a result
    object lacks an attribute or telemetry collection raised.
    """
    try:
        _capture_result_observability(handle, result)
    except Exception as e:
        logger.warning("Failed to capture subagent observability: %s", e)


def _format_chat_trace_result(output: str, chat_trace_id: str) -> str:
    """Append a compact chat trace identifier to a subagent result."""
    return f"{output}\n\nChat Trace ID: {chat_trace_id}"


def _preview_result(result: str, task_id: str, max_chars: int | None) -> str:
    """Render a task result for `wait_tasks`, marking truncation explicitly.

    A silent cut reads to the orchestrator like a subagent that stopped
    mid-sentence, so it "recovers" by re-delegating the same work. The marker
    states that the cut is ours, that the stored answer is complete, and which
    tool returns the untruncated text.

    Args:
        result: The subagent's full result text.
        task_id: Task ID the orchestrator passes to `check_task` for the rest.
        max_chars: Preview budget in characters; `None` disables truncation.

    Returns:
        The result unchanged when it fits the budget, otherwise the first
        `max_chars` characters followed by the truncation marker.
    """
    if max_chars is None or len(result) <= max_chars:
        return result
    return (
        f"{result[:max_chars]}\n\n"
        f"[Result truncated for display: showing {max_chars} of {len(result)} characters. "
        f"The subagent's answer is complete and stored in full. "
        f"Call check_task('{task_id}') to read all of it.]"
    )


async def _drain_steering_messages(message_bus: InMemoryMessageBus, agent_id: str) -> list[str]:
    """Drain pending parent -> child steering messages for a running subagent.

    Pulls everything currently queued for ``agent_id`` and returns the text of
    each ``TASK_UPDATE`` (the message type emitted by ``send_message_to_subagent``).
    Other message types on the queue (e.g. an unused ``CANCEL_REQUEST`` — soft
    cancel runs off the cancel event, not the bus) are ignored, as are empty
    payloads.

    Args:
        message_bus: The bus the running subagent is registered on.
        agent_id: The subagent's bus id (``subagent-{task_id}``).

    Returns:
        Steering instructions in delivery order (may be empty).
    """
    pending = await message_bus.get_messages(agent_id, timeout=0)
    steering: list[str] = []
    for msg in pending:
        if msg.type != MessageType.TASK_UPDATE:
            continue
        payload = msg.payload
        text = payload.get("message") if isinstance(payload, dict) else payload
        if text:
            steering.append(str(text))
    return steering


def _create_general_purpose_config() -> SubAgentConfig:
    """Create the default general-purpose subagent config."""
    return SubAgentConfig(
        name="general-purpose",
        description=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
        instructions=SUBAGENT_SYSTEM_PROMPT,
        can_ask_questions=True,
    )


def _compile_subagent(
    config: SubAgentConfig,
    default_model: str | Model,
) -> CompiledSubAgent:
    """Compile a subagent configuration into a ready-to-use agent.

    Agent resolution priority:
    1. `config["agent"]` — pre-built agent instance, used as-is
    2. `config["agent_factory"]` — callable(config) -> agent
    3. Default — creates `pydantic_ai.Agent` from config fields

    Args:
        config: The subagent configuration.
        default_model: Default model to use if not specified in config.

    Returns:
        CompiledSubAgent with agent instance.
    """
    # 1. Pre-built agent — use as-is
    if config.get("agent") is not None:
        return CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            agent=config["agent"],
            config=config,
        )

    # 2. Agent factory — call it
    factory = config.get("agent_factory")
    if factory is not None:
        custom_agent = factory(config)
        return CompiledSubAgent(
            name=config["name"],
            description=config["description"],
            agent=custom_agent,
            config=config,
        )

    # 3. Default: create plain pydantic-ai Agent
    model = config.get("model", default_model)

    toolsets: list[Any] = []
    ask_parent_toolset = _create_ask_parent_toolset()
    toolsets.append(ask_parent_toolset)

    if config.get("toolsets"):
        toolsets.extend(config["toolsets"])

    agent_kwargs = config.get("agent_kwargs", {})

    agent: Agent[Any, str] = Agent(
        model,
        system_prompt=config["instructions"],
        toolsets=toolsets,
        **agent_kwargs,
    )

    return CompiledSubAgent(
        name=config["name"],
        description=config["description"],
        agent=agent,
        config=config,
    )


def _create_ask_parent_toolset() -> FunctionToolset[Any]:
    """Create toolset with ask_parent tool for subagent communication."""
    toolset: FunctionToolset[Any] = FunctionToolset(id="ask_parent")

    @toolset.tool
    async def ask_parent(ctx: RunContext[Any], question: str) -> str:
        """Ask the parent agent a question and wait for the answer.

        Use this when you need clarification or additional information
        to complete your task. Keep questions specific and actionable.

        Args:
            ctx: The run context.
            question: The question to ask the parent.

        Returns:
            The parent's answer.
        """
        # Try _subagent_state on deps (async mode)
        state = getattr(ctx.deps, "_subagent_state", None)
        if state is not None:
            ask_callback = state.get("ask_callback")
            if ask_callback:
                result: str = await ask_callback(question)
                return result

            _task_manager = state.get("task_manager")
            _task_id = state.get("task_id")

            if _task_manager and _task_id:
                handle = _task_manager.get_handle(_task_id)
                if handle is not None:
                    # Set question on handle so parent can see it via check_task
                    handle.pending_question = question
                    handle.status = TaskStatus.WAITING_FOR_ANSWER

                    # Create a future and wait for answer_subagent to resolve it
                    loop = asyncio.get_running_loop()
                    answer_future: asyncio.Future[str] = loop.create_future()
                    _task_manager.set_answer_future(_task_id, answer_future)

                    try:
                        answer = await asyncio.wait_for(answer_future, timeout=300.0)
                        handle.status = TaskStatus.RUNNING
                        handle.pending_question = None
                        return answer
                    except asyncio.TimeoutError:
                        handle.status = TaskStatus.RUNNING
                        handle.pending_question = None
                        _task_manager.clear_answer_future(_task_id)
                        return "Error: Parent did not respond in time"

        # Fallback: use deps.ask_user callback (plan toolset compatibility)
        ask_user = getattr(ctx.deps, "ask_user", None)
        if ask_user:
            return str(await ask_user(question, []))

        return (
            "Error: Cannot ask parent - no communication channel configured. "
            "In sync mode, pass `ask_user=...` to create_subagent_toolset(), "
            "or use mode='async' so the parent can respond via answer_subagent()."
        )

    return toolset


def create_subagent_toolset(  # noqa: C901
    subagents: list[SubAgentConfig] | None = None,
    default_model: str | Model = "openai:gpt-4.1",
    toolsets_factory: ToolsetFactory | None = None,
    include_general_purpose: bool = True,
    max_nesting_depth: int = 0,
    id: str | None = None,
    registry: Any | None = None,
    descriptions: dict[str, str] | None = None,
    ask_user: AskUserCallback | None = None,
    usage_limits: UsageLimits | UsageLimitsFactory | None = None,
    max_chat_traces: int = 100,
    max_task_handles: int = 500,
    max_result_chars: int | None = 2000,
) -> FunctionToolset[Any]:
    """Create a toolset for delegating tasks to subagents.

    This is the main entry point for using the subagent system. It creates
    a toolset with tools for:
    - `task`: Delegate a task to a subagent (sync or async)
    - `check_task`: Check status of an async task
    - `answer_subagent`: Answer a question from a subagent
    - `list_active_tasks`: List all running background tasks
    - `wait_tasks`: Wait for one or more background tasks to finish
    - `soft_cancel_task`: Request cooperative cancellation
    - `hard_cancel_task`: Immediately cancel a task

    Args:
        subagents: List of subagent configurations. If None, only
            general-purpose subagent will be available.
        default_model: Default model for subagents that don't specify one.
        toolsets_factory: Factory function that creates toolsets for subagents.
            Called with deps when running a task.
        include_general_purpose: Whether to include the default general-purpose
            subagent. Set to False if you want only specialized subagents.
        max_nesting_depth: Maximum depth for nested subagents. 0 means
            subagents cannot spawn their own subagents.
        id: Optional toolset ID. Defaults to "subagents".
        descriptions: Optional mapping of tool name to custom description.
            Keys are tool names (task, check_task, answer_subagent,
            list_active_tasks, wait_tasks, soft_cancel_task, hard_cancel_task).
            When provided, the custom description replaces the built-in default.
        ask_user: Optional callback invoked when a subagent calls `ask_parent`
            in sync mode. Receives the question and must return the answer.
            Required for sync-mode subagents with `can_ask_questions=True`;
            without it the subagent gets a configuration error. In async mode
            the parent answers via `answer_subagent` instead.
        usage_limits: Optional pydantic-ai usage limits for delegated subagent
            runs. Pass a `UsageLimits` instance to reuse the same limits for
            every task, or a factory called once per task with the parent run
            context and selected subagent config. A factory may return `None`
            to run that task without explicit limits. Limits are honoured on
            every retry attempt as well.
        max_chat_traces: Maximum number of chat traces (subagent conversations)
            whose message history is kept in memory for continuation via
            `chat_trace_id`. Least-recently-used traces are evicted past this
            limit; continuing an evicted trace returns an error. Bounds memory
            in long-lived sessions.
        max_task_handles: Maximum number of finished (completed/failed/cancelled)
            task handles retained for status queries and observability. The
            oldest finished handles are evicted past this limit; their token
            usage is folded into `get_total_usage()` totals so aggregates stay
            correct. Bounds memory in long-lived sessions.
        max_result_chars: Character budget for a completed task's result in the
            `wait_tasks` listing, keeping a fan-out of verbose subagents from
            flooding the orchestrator's context. Results past the budget are cut
            and carry an explicit marker pointing at `check_task`, which always
            returns the full text. Pass `None` to never truncate.

    Returns:
        FunctionToolset configured with subagent management tools.

    Raises:
        ValueError: If `max_result_chars` is negative.

    Example:
        ```python
        from pydantic_ai import Agent
        from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

        subagents = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
            ),
        ]

        toolset = create_subagent_toolset(
            subagents=subagents,
            default_model="openai:gpt-4.1",
        )

        agent = Agent("openai:gpt-4.1", toolsets=[toolset])
        ```
    """
    if max_result_chars is not None and max_result_chars < 0:
        raise ValueError(f"max_result_chars must be >= 0 or None, got {max_result_chars}")

    _descs = descriptions or {}

    # Build subagent configs
    configs: list[SubAgentConfig] = list(subagents) if subagents else []
    if include_general_purpose:
        configs.append(_create_general_purpose_config())

    # Compile subagents
    compiled: dict[str, CompiledSubAgent] = {}
    for config in configs:
        compiled[config["name"]] = _compile_subagent(config, default_model)

    # Create shared state
    message_bus = InMemoryMessageBus()
    task_manager = TaskManager(message_bus=message_bus)
    # LRU-ordered: oldest chat trace first, evicted past max_chat_traces.
    message_history_store: OrderedDict[tuple[str, str], list[Any]] = OrderedDict()
    # Chat traces with a task currently running — a trace must finish before
    # it can be continued, otherwise concurrent saves would lose history.
    active_chat_traces: set[tuple[str, str]] = set()
    # Usage from evicted handles, so get_total_usage() survives eviction.
    evicted_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    _terminal_statuses = (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    def evict_finished_handles() -> None:
        """Drop the oldest finished handles past `max_task_handles`.

        Running/waiting tasks are never evicted. Evicted usage is accumulated
        into `evicted_usage` so `get_total_usage()` stays correct.
        """
        finished = [h for h in task_manager.handles.values() if h.status in _terminal_statuses]
        overflow = len(finished) - max_task_handles
        if overflow <= 0:
            return
        finished.sort(key=lambda h: h.completed_at or h.created_at)
        for old in finished[:overflow]:
            if old.usage is not None:
                evicted_usage["input_tokens"] += getattr(old.usage, "input_tokens", 0)
                evicted_usage["output_tokens"] += getattr(old.usage, "output_tokens", 0)
                evicted_usage["requests"] += getattr(old.usage, "requests", 0)
            task_manager.handles.pop(old.task_id, None)

    # Build dynamic task description with available subagents
    subagent_list = "\n".join(f"- {name}: {c.description}" for name, c in compiled.items())
    task_description = _descs.get(
        "task",
        TASK_TOOL_DESCRIPTION.rstrip() + f"\n\nAvailable subagent types:\n{subagent_list}",
    )

    toolset: FunctionToolset[Any] = FunctionToolset(id=id or "subagents")

    @toolset.tool(description=task_description)
    async def task(
        ctx: RunContext[SubAgentDepsProtocol],
        description: str,
        subagent_type: str,
        mode: ExecutionMode = "sync",
        priority: TaskPriority = TaskPriority.NORMAL,
        complexity: Literal["simple", "moderate", "complex"] | None = None,
        requires_user_context: bool = False,
        may_need_clarification: bool = False,
        chat_trace_id: str | None = None,
    ) -> str:
        """Delegate a task to a specialized subagent.

        Args:
            ctx: The run context with dependencies.
            description: Detailed description of the task to perform.
            subagent_type: Name of the subagent to use.
            mode: Execution mode - "sync" (blocking), "async" (background), or "auto".
            priority: Task priority level (for async tasks).
            complexity: Override complexity estimate ("simple", "moderate", "complex").
            requires_user_context: Whether task needs ongoing user interaction.
            may_need_clarification: Whether task might need clarifying questions.
            chat_trace_id: Optional explicit chat trace ID. When omitted, a new subagent
                conversation is created. When provided, this subagent resumes from
                the previous successful task with the same chat trace.
        """
        # Validate subagent_type — check static compiled dict first, then dynamic registry
        if subagent_type in compiled:
            subagent = compiled[subagent_type]
        elif (
            registry is not None
            and hasattr(registry, "get_compiled")
            and registry.get_compiled(subagent_type)
        ):
            subagent = registry.get_compiled(subagent_type)
        else:
            # Build available list from both sources
            available_names = list(compiled.keys())
            if registry is not None and hasattr(registry, "list_agents"):
                available_names.extend(registry.list_agents())
            available = ", ".join(available_names)
            return f"Error: Unknown subagent '{subagent_type}'. Available: {available}"

        config = subagent.config
        agent = subagent.agent

        if agent is None:
            return f"Error: Subagent '{subagent_type}' is not properly initialized"

        resolved_usage_limits = (
            usage_limits(ctx, config) if callable(usage_limits) else usage_limits
        )

        # Create deps for subagent
        parent_deps = ctx.deps
        subagent_deps = parent_deps.clone_for_subagent(max_nesting_depth - 1)

        # Build runtime toolsets from factory if provided
        runtime_toolsets = toolsets_factory(subagent_deps) if toolsets_factory else None

        # Generate task ID
        task_id = str(uuid.uuid4())[:8]

        effective_chat_trace_id = chat_trace_id or uuid.uuid4().hex
        chat_trace_key = (config["name"], effective_chat_trace_id)
        if chat_trace_key in active_chat_traces:
            return (
                f"Error: chat trace '{effective_chat_trace_id}' already has a running "
                f"task on subagent '{config['name']}'. Wait for it to finish "
                f"(check_task/wait_tasks) before continuing this conversation."
            )
        message_history = message_history_store.get(chat_trace_key)
        if message_history is not None:
            # Refresh LRU recency so actively-continued traces aren't evicted.
            message_history_store.move_to_end(chat_trace_key)
        elif chat_trace_id is not None:
            return (
                f"Error: no saved conversation for chat_trace_id '{chat_trace_id}' "
                f"with subagent '{config['name']}' (unknown, evicted, or its first "
                f"run failed). Omit chat_trace_id to start a new conversation."
            )

        def save_message_history(messages: list[Any]) -> None:
            message_history_store[chat_trace_key] = messages
            message_history_store.move_to_end(chat_trace_key)
            while len(message_history_store) > max_chat_traces:
                message_history_store.popitem(last=False)

        # Bound retained finished handles before registering a new one.
        evict_finished_handles()

        # Resolve mode if "auto"
        if mode == "auto":
            characteristics = TaskCharacteristics(
                estimated_complexity=complexity or config.get("typical_complexity", "moderate"),
                requires_user_context=requires_user_context
                or config.get("typically_needs_context", False),
                may_need_clarification=may_need_clarification,
            )
            resolved_mode = decide_execution_mode(characteristics, config)
        else:
            resolved_mode = mode

        if resolved_mode == "sync":
            handle = TaskHandle(
                task_id=task_id,
                subagent_name=config["name"],
                description=description,
                status=TaskStatus.RUNNING,
                priority=priority,
                chat_trace_id=effective_chat_trace_id,
                started_at=datetime.now(),
            )
            task_manager.handles[task_id] = handle
            active_chat_traces.add(chat_trace_key)
            try:
                result = await _run_sync(
                    agent=agent,
                    config=config,
                    description=description,
                    deps=subagent_deps,
                    task_id=task_id,
                    extra_toolsets=runtime_toolsets,
                    ask_user=ask_user,
                    usage_limits=resolved_usage_limits,
                    handle=handle,
                    message_history=message_history,
                    on_message_history=save_message_history,
                )
            finally:
                active_chat_traces.discard(chat_trace_key)
            # Don't advertise continuation when the run failed and nothing was
            # ever saved for this trace — the chat_trace_id would resume nothing.
            if handle.status != TaskStatus.FAILED or chat_trace_key in message_history_store:
                return _format_chat_trace_result(result, effective_chat_trace_id)
            return result
        else:
            active_chat_traces.add(chat_trace_key)
            try:
                return await _run_async(
                    agent=agent,
                    config=config,
                    description=description,
                    deps=subagent_deps,
                    task_id=task_id,
                    task_manager=task_manager,
                    message_bus=message_bus,
                    extra_toolsets=runtime_toolsets,
                    priority=priority,
                    usage_limits=resolved_usage_limits,
                    chat_trace_id=effective_chat_trace_id,
                    message_history=message_history,
                    on_message_history=save_message_history,
                    # The background task owns the trace until it finishes.
                    on_run_finished=lambda: active_chat_traces.discard(chat_trace_key),
                )
            except BaseException:
                # _run_async failed before the background task took ownership.
                active_chat_traces.discard(chat_trace_key)
                raise

    @toolset.tool(description=_descs.get("check_task", CHECK_TASK_DESCRIPTION))
    async def check_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Check the status of a background task.

        Args:
            ctx: The run context.
            task_id: The task ID returned when the task was started.
        """
        handle = task_manager.get_handle(task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        status_info = [
            f"Task: {task_id}",
            f"Subagent: {handle.subagent_name}",
            f"Status: {handle.status}",
            f"Description: {handle.description}",
        ]
        # Only advertise continuation for completed tasks — a failed or still
        # running task has not saved this run's history yet (matches wait_tasks).
        if handle.chat_trace_id is not None and handle.status == TaskStatus.COMPLETED:
            status_info.append(f"Chat Trace ID: {handle.chat_trace_id}")

        if handle.status == TaskStatus.COMPLETED:
            status_info.append(f"Result: {handle.result}")
        elif handle.status == TaskStatus.FAILED:
            status_info.append(f"Error: {handle.error}")
        elif handle.status == TaskStatus.WAITING_FOR_ANSWER:
            status_info.append(f"Question: {handle.pending_question}")
        elif handle.started_at:
            elapsed = (datetime.now() - handle.started_at).total_seconds()
            status_info.append(f"Running for: {elapsed:.1f}s")

        return "\n".join(status_info)

    @toolset.tool(description=_descs.get("answer_subagent", ANSWER_SUBAGENT_DESCRIPTION))
    async def answer_subagent(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
        answer: str,
    ) -> str:
        """Answer a question from a subagent.

        Args:
            ctx: The run context.
            task_id: The task ID of the waiting subagent.
            answer: Your answer to the subagent's question.
        """
        handle = task_manager.get_handle(task_id)
        if handle is None:
            return f"Error: Task '{task_id}' not found"

        if handle.status != TaskStatus.WAITING_FOR_ANSWER:
            return f"Error: Task '{task_id}' is not waiting for an answer (status: {handle.status})"

        # Resolve the answer future that ask_parent is waiting on
        future = task_manager.get_answer_future(task_id)
        if future is not None and not future.done():
            future.set_result(answer)
            return f"Answer sent to task '{task_id}'"

        return "Error: Could not send answer - subagent is no longer waiting"

    @toolset.tool(
        description=_descs.get("send_message_to_subagent", SEND_MESSAGE_TO_SUBAGENT_DESCRIPTION)
    )
    async def send_message_to_subagent(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
        message: str,
    ) -> str:
        """Steer a running async subagent with an unprompted message.

        The message is queued for the subagent and folded into its next model
        request as an extra user instruction, so it adapts without losing
        partial progress. Works only while the task is still running.

        Args:
            ctx: The run context.
            task_id: The task ID of the running async subagent.
            message: The steering instruction to deliver.
        """
        agent_id = f"subagent-{task_id}"
        if not message_bus.is_registered(agent_id):
            handle = task_manager.get_handle(task_id)
            if handle is None:
                return f"Error: Task '{task_id}' not found"
            return (
                f"Error: Task '{task_id}' is not accepting messages "
                f"(status: {handle.status}). Steering only works for running "
                "async tasks."
            )

        await message_bus.send(
            AgentMessage(
                type=MessageType.TASK_UPDATE,
                sender="parent",
                receiver=agent_id,
                payload={"message": message},
                task_id=task_id,
            )
        )
        return (
            f"Message delivered to task '{task_id}'; "
            "it will be applied on the subagent's next step."
        )

    @toolset.tool(description=_descs.get("list_active_tasks", LIST_ACTIVE_TASKS_DESCRIPTION))
    async def list_active_tasks(
        ctx: RunContext[SubAgentDepsProtocol],
    ) -> str:
        """List all active background tasks."""
        active_ids = task_manager.list_active_tasks()

        if not active_ids:
            return "No active background tasks."

        lines = ["Active background tasks:"]
        for tid in active_ids:
            handle = task_manager.get_handle(tid)
            if handle:  # pragma: no branch
                desc = handle.description[:50]
                lines.append(f"- {tid}: {handle.subagent_name} ({handle.status}) - {desc}...")

        return "\n".join(lines)

    @toolset.tool(description=_descs.get("wait_tasks", WAIT_TASKS_DESCRIPTION))
    async def wait_tasks(
        ctx: RunContext[SubAgentDepsProtocol],
        task_ids: list[str],
        timeout: float = 300.0,
        mode: Literal["all", "any"] = "all",
    ) -> str:
        """Wait for multiple background tasks to complete.

        Args:
            ctx: The run context.
            task_ids: List of task IDs to wait for.
            timeout: Maximum seconds to wait (default 300s / 5 minutes).
            mode: `"all"` (default) waits for every task to finish.
                `"any"` returns as soon as one task reaches a terminal
                state (completed, failed, or cancelled), so the orchestrator
                can react to the first finisher without stalling on the
                slowest one.
        """
        # Collect asyncio.Task objects for the requested task_ids
        tasks_to_await: list[tuple[str, asyncio.Task[Any]]] = []
        for tid in task_ids:
            t = task_manager.tasks.get(tid)
            if t is not None and not t.done():
                tasks_to_await.append((tid, t))

        if tasks_to_await:
            aws = [t for _, t in tasks_to_await]
            # Both modes route through `asyncio.wait`. Unlike
            # `asyncio.wait_for(asyncio.gather(...))`, `asyncio.wait` does
            # *not* cascade cancellation to its constituent tasks — neither on
            # timeout nor when its caller is cancelled (e.g. pydantic-ai's
            # `_call_tools` sibling-cancel hitting this tool call). Workers
            # keep owning their lifecycle, which is what an orchestrator
            # expects.
            return_when = asyncio.FIRST_COMPLETED if mode == "any" else asyncio.ALL_COMPLETED
            await asyncio.wait(aws, timeout=timeout, return_when=return_when)

        # Collect results
        lines: list[str] = []
        finished_count = 0
        for tid in task_ids:
            handle = task_manager.get_handle(tid)
            if handle is None:
                lines.append(f"- {tid}: not found")
                continue
            status = handle.status
            if status == "completed":
                finished_count += 1
                result_preview = _preview_result(handle.result or "", tid, max_result_chars)
                chat_trace_line = ""
                if handle.chat_trace_id is not None:
                    chat_trace_line = f"Chat Trace ID: {handle.chat_trace_id}\n"
                lines.append(
                    f"- {tid} ({handle.subagent_name}): COMPLETED\n"
                    f"{chat_trace_line}{result_preview}"
                )
            elif status == "failed":
                finished_count += 1
                lines.append(f"- {tid} ({handle.subagent_name}): FAILED - {handle.error}")
            elif status == "cancelled":
                finished_count += 1
                lines.append(f"- {tid} ({handle.subagent_name}): CANCELLED - {handle.error}")
            else:
                lines.append(f"- {tid} ({handle.subagent_name}): {status}")

        total = len(task_ids)
        still_running = total - finished_count
        header_parts = [f"mode={mode}", f"{finished_count}/{total} finished"]
        if still_running > 0:
            header_parts.append(f"{still_running} still running")
        header = f"Task results ({', '.join(header_parts)}):"

        return header + "\n" + "\n\n".join(lines)

    @toolset.tool(description=_descs.get("soft_cancel_task", SOFT_CANCEL_TASK_DESCRIPTION))
    async def soft_cancel_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Request cooperative cancellation of a background task.

        Args:
            ctx: The run context.
            task_id: The task to cancel.
        """
        success = await task_manager.soft_cancel(task_id)
        if success:
            return f"Cancellation requested for task '{task_id}'"
        return f"Error: Task '{task_id}' not found"

    @toolset.tool(description=_descs.get("hard_cancel_task", HARD_CANCEL_TASK_DESCRIPTION))
    async def hard_cancel_task(
        ctx: RunContext[SubAgentDepsProtocol],
        task_id: str,
    ) -> str:
        """Immediately cancel a background task.

        Args:
            ctx: The run context.
            task_id: The task to cancel.
        """
        success = await task_manager.hard_cancel(task_id)
        if success:
            return f"Task '{task_id}' has been cancelled"
        return f"Error: Task '{task_id}' not found"

    # Expose task_manager for external monitoring (e.g., push notifications)
    toolset.task_manager = task_manager  # type: ignore[attr-defined]
    toolset.message_history_store = message_history_store  # type: ignore[attr-defined]

    def get_total_usage() -> dict[str, int]:
        """Get aggregate token usage across all completed subagent tasks.

        Returns dict with `input_tokens`, `output_tokens`, `total_tokens`, `requests`.
        """
        totals: dict[str, int] = {
            "input_tokens": evicted_usage["input_tokens"],
            "output_tokens": evicted_usage["output_tokens"],
            "total_tokens": 0,
            "requests": evicted_usage["requests"],
        }
        for handle in task_manager.list_handles():
            if handle.usage is not None:
                totals["input_tokens"] += getattr(handle.usage, "input_tokens", 0)
                totals["output_tokens"] += getattr(handle.usage, "output_tokens", 0)
                totals["requests"] += getattr(handle.usage, "requests", 0)
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
        return totals

    toolset.get_total_usage = get_total_usage  # type: ignore[attr-defined]

    return toolset


async def _run_sync(
    agent: Any,
    config: SubAgentConfig,
    description: str,
    deps: Any,
    task_id: str,
    extra_toolsets: list[Any] | None = None,
    ask_user: AskUserCallback | None = None,
    usage_limits: UsageLimits | None = None,
    handle: TaskHandle | None = None,
    message_history: list[Any] | None = None,
    on_message_history: Callable[[list[Any]], None] | None = None,
) -> str:
    """Run a subagent task synchronously (blocking).

    Args:
        agent: The pydantic-ai Agent instance.
        config: Subagent configuration.
        description: Task description.
        deps: Dependencies for the subagent.
        task_id: Unique task identifier.
        extra_toolsets: Additional toolsets to pass to agent.run().
        ask_user: Optional callback for `ask_parent` in sync mode. When
            provided, it is attached to the cloned subagent deps via
            `_subagent_state["ask_callback"]` so `ask_parent` resolves to
            it. The parent agent cannot answer directly in sync mode because
            its run loop is blocked here.
        usage_limits: Optional pydantic-ai usage limits forwarded to the
            subagent run (honoured on every retry attempt).
        handle: Optional task handle populated for Python-side observability.
        message_history: Optional prior message history for a subagent chat trace.
        on_message_history: Optional callback that receives the successful
            run's full message history.

    Returns:
        The subagent's response.
    """
    can_ask = config.get("can_ask_questions", True)
    max_questions = config.get("max_questions")

    if ask_user is not None:
        # Reuse the async-mode `_subagent_state["ask_callback"]` path so
        # `ask_parent` has a single resolution order across both modes.
        deps._subagent_state = {"ask_callback": ask_user}

    prompt = get_task_instructions_prompt(
        description,
        can_ask_questions=can_ask,
        max_questions=max_questions,
    )

    run_kwargs: dict[str, Any] = {"deps": deps}
    if extra_toolsets:
        run_kwargs["toolsets"] = extra_toolsets
    if usage_limits is not None:
        run_kwargs["usage_limits"] = usage_limits
    if message_history is not None:
        run_kwargs["message_history"] = message_history
    if handle is not None and handle.chat_trace_id is not None:
        run_kwargs["conversation_id"] = handle.chat_trace_id

    try:
        result = await run_with_retry(
            agent,
            prompt,
            run_kwargs=run_kwargs,
            retry=RetryConfig.from_config(config),
        )
        _capture_message_history(result, on_message_history)
        output = _serialize_output(result.output)
        if handle is not None:
            handle.result = output
            handle.error = None
            handle.status = TaskStatus.COMPLETED
            handle.completed_at = datetime.now()
            # Telemetry is best-effort and runs after the run is marked complete,
            # so a capture failure can never flip a successful run to FAILED.
            _capture_observability_best_effort(handle, result)
        return output
    except Exception as e:
        if handle is not None:
            handle.status = TaskStatus.FAILED
            handle.error = str(e)
            handle.completed_at = datetime.now()
        return f"Error executing task: {e}"


async def _run_async(
    agent: Any,
    config: SubAgentConfig,
    description: str,
    deps: Any,
    task_id: str,
    task_manager: TaskManager,
    message_bus: InMemoryMessageBus,
    priority: TaskPriority = TaskPriority.NORMAL,
    extra_toolsets: list[Any] | None = None,
    usage_limits: UsageLimits | None = None,
    chat_trace_id: str | None = None,
    message_history: list[Any] | None = None,
    on_message_history: Callable[[list[Any]], None] | None = None,
    on_run_finished: Callable[[], None] | None = None,
) -> str:
    """Run a subagent task asynchronously (background).

    Args:
        agent: The pydantic-ai Agent instance.
        config: Subagent configuration.
        description: Task description.
        deps: Dependencies for the subagent.
        task_id: Unique task identifier.
        task_manager: Task manager for tracking.
        message_bus: Message bus for communication.
        priority: Task priority level.
        extra_toolsets: Additional toolsets to pass to agent.run().
        usage_limits: Optional pydantic-ai usage limits forwarded to the
            subagent run (honoured on every retry attempt).
        chat_trace_id: Optional chat trace ID for continuing this subagent
            conversation.
        message_history: Optional prior message history for a subagent chat trace.
        on_message_history: Optional callback that receives the successful
            run's full message history.
        on_run_finished: Optional callback invoked exactly once when the
            background run finishes (completed, failed, or cancelled). Used to
            release the chat trace for continuation.

    Returns:
        Task handle information as string.
    """
    # Create task handle
    handle = TaskHandle(
        task_id=task_id,
        subagent_name=config["name"],
        description=description,
        status=TaskStatus.PENDING,
        priority=priority,
        chat_trace_id=chat_trace_id,
    )

    # Register subagent for messaging
    agent_id = f"subagent-{task_id}"
    try:
        message_bus.register_agent(agent_id)
    except ValueError:
        pass  # Already registered

    # Inject _subagent_state on deps so ask_parent can communicate with parent
    deps._subagent_state = {
        "task_manager": task_manager,
        "task_id": task_id,
    }

    async def run_task() -> None:
        """Execute the task and update handle."""
        can_ask = config.get("can_ask_questions", True)
        max_questions = config.get("max_questions")

        prompt = get_task_instructions_prompt(
            description,
            can_ask_questions=can_ask,
            max_questions=max_questions,
        )

        run_kwargs: dict[str, Any] = {"deps": deps}
        if extra_toolsets:
            run_kwargs["toolsets"] = extra_toolsets
        if usage_limits is not None:
            run_kwargs["usage_limits"] = usage_limits
        if message_history is not None:
            run_kwargs["message_history"] = message_history
        if chat_trace_id is not None:
            run_kwargs["conversation_id"] = chat_trace_id

        def _on_retry(attempt: int, exc: BaseException, delay: float) -> None:
            handle.status = TaskStatus.RETRYING
            handle.retry_count = attempt
            handle.error = f"Transient error (retry {attempt}): {exc}"

        def _cancel_requested() -> bool:
            # Cooperative (soft) cancellation: TaskManager.soft_cancel sets this
            # event, and run_with_retry polls it between graph nodes so the
            # subagent stops at a clean boundary. Raised CancelledError is
            # handled by the `except asyncio.CancelledError` branch below.
            cancel_event = task_manager.get_cancel_event(task_id)
            return cancel_event is not None and cancel_event.is_set()

        async def _pending_steering() -> list[str]:
            return await _drain_steering_messages(message_bus, agent_id)

        try:
            result = await run_with_retry(
                agent,
                prompt,
                run_kwargs=run_kwargs,
                retry=RetryConfig.from_config(config),
                on_retry=_on_retry,
                cancel_check=_cancel_requested,
                inject_messages=_pending_steering,
            )
            _capture_message_history(result, on_message_history)
            handle.result = _serialize_output(result.output)
            handle.error = None
            handle.status = TaskStatus.COMPLETED
            # Telemetry is best-effort and runs after the run is marked complete,
            # so a capture failure can never flip a successful run to FAILED.
            _capture_observability_best_effort(handle, result)
        except asyncio.CancelledError:
            handle.status = TaskStatus.CANCELLED
            handle.error = "Task was cancelled"
        except Exception as e:
            handle.status = TaskStatus.FAILED
            handle.error = str(e)
        finally:
            handle.completed_at = datetime.now()
            message_bus.unregister_agent(agent_id)
            task_manager.clear_answer_future(task_id)
            task_manager.cleanup_task(task_id)
            if on_run_finished is not None:
                on_run_finished()

    # Create the background task
    task_manager.create_task(task_id, run_task(), handle)

    response = f"Task started in background.\nTask ID: {task_id}\nSubagent: {config['name']}\n"
    if chat_trace_id is not None:
        response += f"Chat Trace ID: {chat_trace_id}\n"
    response += f"Use check_task('{task_id}') to check status."
    return response


# Alias for backwards compatibility
SubAgentToolset = create_subagent_toolset
