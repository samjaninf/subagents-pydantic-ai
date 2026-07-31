# Refactor plan: correctness, code quality, documentation

Status: proposed
Scope decision: **source-compatible only.** `create_subagent_toolset()`, `SubAgentToolset`,
`toolset.task_manager`, `_compile_subagent`, and every current export keep working, because
`pydantic-deep` imports `_compile_subagent`, patches `subagents_pydantic_ai.toolset.Agent` in its
tests, and reads `getattr(subagent_toolset, "task_manager", None)` in `pydantic_deep/agent.py`,
`apps/cli/agent.py`, and `apps/deepresearch/src/deepresearch/app.py`.
Message bus decision: the public bus surface (`InMemoryMessageBus`, `create_message_bus`,
`MessageBusProtocol`) stays; only the steering internals move onto pydantic-ai's `enqueue`.

Reference repos for target quality: `pydantic-ai-harness` (package shape, typing bar, writing
style, `agent_docs/`) and `pydantic-ai-backend` (docs information architecture, mkdocs nav).

## 1. Baseline as measured

Measured at `3f217f4` (v0.2.11), pydantic-ai 2.0.0, Python 3.14 local / 3.10–3.13 in CI:

| Check | Result |
|---|---|
| `pytest` | 396 passed |
| `coverage` | 100.00% statements and branches |
| `ruff format --check`, `ruff check` | clean |
| `pyright` (basic mode, `src` only) | 0 errors |
| `mypy src/` (strict) | clean |
| `make typecheck-mypy` (`src` + `tests`) | **583 errors — target is broken** |
| src size | 11 modules, ~3.9k lines |
| docs size | 30 pages, ~6.5k lines |

The headline number is misleading. 100% branch coverage coexists with a defect that changes what
every model sees on every `check_task` call (B1 below), because no test asserts the tool's rendered
output. Coverage is measuring line execution, not behaviour.

Two config choices hide the rest: `pyright` runs in `basic` mode with `reportCallIssue`,
`reportAssignmentType`, `reportUnknownMemberType`, `reportPrivateUsage`, and
`reportTypedDictNotRequiredAccess` all disabled, and CI runs `mypy` only against `src/`. Harness
runs pyright **strict** with no per-rule escapes.

## 2. Defects

Ranked by user impact. Every entry has a concrete failure, not a style objection.

### B1 — Enum leaks into model-facing text (Python 3.11+)

`TaskStatus` is a `str`-mixin `Enum`. Since Python 3.11, `format()` on a mixin enum returns
`"TaskStatus.COMPLETED"`, not `"completed"`. Reproduced on 3.14:

```python
f"Status: {handle.status}"   # -> 'Status: TaskStatus.WAITING_FOR_ANSWER'
```

Affected model-facing output: `toolset.py:992` (`check_task`), `toolset.py:1099`
(`list_active_tasks`), `toolset.py:1167` (`wait_tasks`, non-terminal branch). The orchestrating
model is told the status is `TaskStatus.WAITING_FOR_ANSWER` while the tool descriptions and docs
promise `waiting_for_answer`. On Python 3.10 the same code renders `completed`, so behaviour
differs across the supported matrix.

Fix: render `handle.status.value` (or convert to `StrEnum` on 3.11+ with a 3.10 shim). Add tests
that assert the exact tool return strings, not just that a branch executed.

### B2 — `except Exception` swallows pydantic-ai control-flow signals

`_run_sync` (`toolset.py:1317`) and `run_task` (`toolset.py:1447`) catch `Exception` and convert it
into a normal string result (`f"Error executing task: {e}"`). That catch also swallows
`CallDeferred`, `ApprovalRequired`, `SkipModelRequest`, `SkipToolValidation`, `SkipToolExecution`,
and `UserError`.

Consequences: a subagent whose toolset uses deferred tools or human-in-the-loop approval cannot
work at all — the signal that is supposed to suspend the run becomes a string the parent model
reads as a completed task. A `UserError` (a setup bug no retry can fix) is reported to the model as
a task failure. And a genuine child failure is presented to the parent as a *successful* tool call
whose content happens to start with `Error`, so pydantic-ai's retry budget never engages.

Fix, mirroring `pydantic_ai_harness/subagents/_toolset.py`: an `_ALWAYS_PROPAGATE` tuple that
re-raises control-flow signals and `UserError`, and `ModelRetry` for genuine child failures. Add
`on_failure` (soft steering message) and `contain_errors` knobs so applications can opt into the
current containment behaviour explicitly.

This is the plan's one intentional behaviour change. It is additive at the API level, but the
string a parent model receives on child failure changes.

### B3 — Background tasks outlive their parent run

`_run_async` spawns `asyncio.create_task(run_task())` and nothing cancels it when the parent agent
run ends. An orphan keeps executing against deps the application has already torn down, and one
blocked in `ask_parent` sits on a future nobody will resolve for the full 300 s timeout.

Harness guarantees the opposite explicitly: task state is keyed by `ctx.run_id` and a `wrap_run`
finalizer ensures no task outlives its parent run.

Fix inside the compatibility budget: add `TaskManager.cancel_all(run_id)` and call it from a
`wrap_run` finalizer on `SubAgentCapability`. Keep `TaskManager.handles` globally readable so
`deepresearch/app.py`'s iteration keeps working.

### B4 — Task ids are not isolated per run

`TaskManager` lives for the lifetime of the toolset, and the toolset is built once per agent
(`pydantic_deep/agent.py` does exactly this). `check_task`, `hard_cancel_task`, `soft_cancel_task`,
`answer_subagent`, and `send_message_to_subagent` accept any `task_id` in that shared map. In a
server that serves several users from one agent instance, run A can inspect, answer, or cancel run
B's task.

Fix: record `run_id` on `TaskHandle` and filter lookups in the tools by `ctx.run_id`, while leaving
`TaskManager.handles` unfiltered for the downstream observability contract.

### B5 — `hard_cancel` overwrites a completed task's outcome

`message_bus.py:452-465` guards with `if not task.done()`. A task that already set
`status = COMPLETED` and is executing its `finally` block is still not `done()`, so `hard_cancel`
overwrites the real result with `CANCELLED` and moves `completed_at`. The code comment
acknowledges the race; the guard does not close it.

Fix: an idempotent terminal transition — first terminal status wins — as in `TaskState.finish()`
in harness.

### B6 — Private attribute written onto the caller's deps object

`toolset.py:1281` and `toolset.py:1385` do `deps._subagent_state = {...}`. The deps class is the
application's, and `SubAgentDepsProtocol` places no constraint on it. A deps dataclass declared
`frozen=True` or `slots=True` — both idiomatic, and nothing in the docs warns against them —
raises `AttributeError` at the first delegation. The state itself is an untyped
`dict[str, Any]` read back through `state.get("ask_callback")`.

Fix: a typed `_SubAgentState` dataclass carried through a `contextvars.ContextVar` instead of
attribute injection, with the attribute path kept as a documented fallback for one release.

### B7 — `ask_parent` timeout is a hardcoded 300 s

`toolset.py:371`. Not configurable at any level. Harness exposes `ask_timeout_seconds`.
Fix: a `ask_timeout_seconds` parameter on the toolset and capability, default 300.0.

### B8 — Naive local-time timestamps

`datetime.now()` with no tzinfo in `types.py:286,321`, `toolset.py:752,1007,1312,1451`,
`message_bus.py:342,465`. `TaskHandle.created_at/started_at/completed_at` are handed to consumers
(pydantic-deep renders them), elapsed time is computed by subtraction, and `evict_finished_handles`
sorts by them. Across a DST transition the arithmetic is wrong and ordering can invert.
Fix: `datetime.now(timezone.utc)` throughout.

### B9 — `_drive_run` has already drifted from the core loop it copies

`retry.py:_drive_run` is a hand copy of `Agent.run`'s internal driving loop and depends on five
private pydantic-ai APIs: `run._advance_graph`, `run._run_node_with_hooks`,
`_agent_graph.build_run_context`, `_agent_graph.ModelRequestNode`, and
`run.ctx.deps.root_capability`.

It has already diverged. Core drains the wrapped stream after the handler returns, unconditionally:

```python
if _handler is not None:
    await _handler(run_ctx, wrapped)
async for _ in wrapped:   # core: always drains
    pass
```

The copy drains only when there is no handler:

```python
if event_stream_handler is not None:
    await event_stream_handler(run_ctx, wrapped)
else:
    async for _ in wrapped:   # copy: else-branch only
        pass
```

So a handler that does not fully consume the stream leaves the node's stream wrappers unfinished on
the retry path but not on the plain path.

The copy cannot simply be deleted — driving via public `run.next(node)` alone loses event streaming,
which is the reason it exists. Plan: realign it with core, pin a `pydantic-ai-slim` lower bound, add
a drift test that asserts the handler fires and the stream is drained on the retry path, and open an
upstream issue asking for a public way to drive a run with a custom step (or for `Agent.run` to
accept a retry policy). This follows harness's core-boundary rule: propose the core primitive rather
than maintaining a second runtime.

### B10 — Steering mutates graph nodes instead of using the public primitive

`retry.py:227-233` appends `UserPromptPart`s directly into `node.request.parts` to deliver parent →
child steering. pydantic-ai 2.0 exposes `AgentRun.enqueue` and `RunContext.enqueue` for exactly
this, and harness uses `run_handle.enqueue(...)`.

Fix: deliver steering through `AgentRun.enqueue`. Per the scope decision the bus stays public and
keeps carrying the messages; only the delivery mechanism changes.

### B11 — `max_nesting_depth` enforces nothing

The only use is `clone_for_subagent(max_nesting_depth - 1)` (`toolset.py:689`). With the default
`0` the library calls the application's clone method with `-1`. Nothing in the library stops a
nested toolset from delegating further; the actual gate is whatever `toolsets_factory` returns.

Meanwhile `README.md` and `docs/index.md` advertise "Nested Hierarchies" and
`create_subagent_toolset`'s docstring states "0 means subagents cannot spawn their own subagents".
`docs/examples/nesting.md` annotates `max_nesting_depth=1  # Allow one level of nesting`, which is
not what the parameter does.

Secondary problem in the same example: it calls `create_subagent_toolset()` *inside*
`toolsets_factory`, so every single delegation allocates a fresh `TaskManager`, message bus, and
chat-trace LRU, and nested async tasks are unreachable from the top-level `task_manager`.

Fix: track delegation depth in the toolset and refuse past the limit, or rename and document the
parameter for what it is. Then correct the docs and the example either way.

### B12 — Smaller correctness items

| # | Item | Location |
|---|---|---|
| B12a | Handler exceptions swallowed with bare `pass` | `message_bus.py:69-73` |
| B12b | `asyncio.get_event_loop()` inside a coroutine (deprecated since 3.10) | `message_bus.py:105` |
| B12c | `except (AssertionError, LookupError)` around cost lookup masks real assertion failures | `toolset.py:170` |
| B12d | `handle.status == "completed"` (string) in `wait_tasks` vs `TaskStatus.COMPLETED` in `check_task` | `toolset.py:1150` vs `997` |
| B12e | `str(uuid.uuid4())[:8]` task ids; a collision silently overwrites a live handle | `toolset.py:699,931` |
| B12f | `import dataclasses, json` inside the function body | `toolset.py:76-77` |
| B12g | `evict_finished_handles()` runs before the new handle is registered, so the map transiently holds `max_task_handles + 1` | `toolset.py:731` |
| B12h | `409 Conflict` and `425 Too Early` classified as transient | `retry.py:46` |

## 3. Quality debt against the reference repos

### D1 — `create_subagent_toolset` is a 17-parameter closure factory with monkey-patched output

840 lines, `# noqa: C901`, and `[tool.ruff.lint.mccabe] max-complexity = 30` with the comment
"Factory functions are intentionally complex" — the config bent to fit the code. Eleven tool bodies
are nested closures over shared mutable state (`compiled`, `message_history_store`,
`active_chat_traces`, `evicted_usage`, `task_manager`), so none of them is individually testable or
typed.

Then three attributes are attached to the returned object after the fact:

```python
toolset.task_manager = task_manager                  # type: ignore[attr-defined]
toolset.message_history_store = message_history_store # type: ignore[attr-defined]
toolset.get_total_usage = get_total_usage            # type: ignore[attr-defined]
```

`toolset.task_manager` is load-bearing downstream, so this is de facto public API expressed as
three type errors.

Fix, source-compatible: promote `SubAgentToolset` from an alias for the function to a real
`FunctionToolset` subclass with typed attributes and methods, and let `create_subagent_toolset()`
return an instance. `create_subagent_toolset(...)`, `SubAgentToolset(subagents=[...])`,
`isinstance(t, FunctionToolset)`, and `toolset.task_manager` all keep working; the three
`type: ignore`s disappear. Harness's `SubAgentToolset(FunctionToolset[AgentDepsT])` is the model.

### D2 — `SubAgentConfig` is a `total=False` TypedDict with three "required" keys

`name`, `description`, and `instructions` are documented as required, but `total=False` makes them
optional to the type checker. The code then indexes them unguarded in roughly a dozen places
(`config["name"]`, `config["description"]`, `config["instructions"]`), so a config missing one
raises `KeyError` deep inside delegation instead of failing at construction.
`reportTypedDictNotRequiredAccess = false` in `pyproject.toml` is what keeps this invisible.

Fix, source-compatible: split into a required base plus a `total=False` extension.

```python
class _SubAgentConfigRequired(TypedDict):
    name: str
    description: str
    instructions: str

class SubAgentConfig(_SubAgentConfigRequired, total=False):
    ...
```

Call sites are unchanged; both type checkers now enforce the three keys.

### D3 — `Any` in public and internal signatures

`CompiledSubAgent.agent: object | None` ("typed as object to avoid circular imports" — `types.py`
does not import `message_bus`, so the circularity does not exist), `TaskHandle.usage: Any`,
`TaskHandle.finish_reason: Any`, `ToolsetFactory = Callable[[Any], list[Any]]`,
`AgentFactory = Callable[[SubAgentConfig], Any]`, `TaskManager.handles: dict[str, Any]`,
`TaskManager.create_task(coro: Any, handle: Any)`, and `_run_sync`/`_run_async`'s `agent: Any`.

`RunUsage`, `FinishReason`, and `TaskHandle` can all be imported under `TYPE_CHECKING`.
`ToolsetFactory` should be generic over deps. `AbstractAgent[AgentDepsT, Any]` is the right type for
the agent slot, as in harness.

### D4 — Dead weight

- `InMemoryMessageBus.ask`, `.answer`, `._pending_questions`, `.add_handler`, `.remove_handler`,
  `.registered_agents` — none is called by the library. Retained per the scope decision, but they
  must be documented as an extension surface rather than presented as part of how subagents work.
- `create_message_bus(backend="memory")` — a factory over exactly one backend that raises for
  anything else.
- `SubAgentDepsProtocol.subagents: dict[str, Any]` — never read by the library. Every application
  is forced to carry a field the library ignores. Cannot be removed under the compatibility budget;
  deprecate in docs and stop requiring it in examples.
- `DUAL_MODE_SYSTEM_PROMPT` — exported, never used internally.
- `get_subagent_system_prompt(include_dual_mode=...)` — the parameter is accepted and ignored.

### D5 — Module layout

Flat modules with everything public. Harness uses a package per feature with `_`-prefixed
implementation modules and an `__init__.py` that defines the public surface. Under the compatibility
budget the current module paths must keep resolving (`subagents_pydantic_ai.toolset._compile_subagent`
is imported by `pydantic_deep/features/teams/toolset.py`), so: split `toolset.py` internally into
`_execution.py`, `_observability.py`, `_chat_trace.py`, and `_tools.py`, and keep `toolset.py` as
the public façade re-exporting the current names.

### D6 — Tooling

| Item | Now | Target |
|---|---|---|
| pyright mode | `basic`, `tests` excluded, 12 rules off | `strict`, tests included, escapes justified per-line |
| mypy on tests | `module = "tests.*"` override never matches (no `tests/__init__.py`), so `make typecheck-mypy` fails with 583 errors | override fixed; target green; run in CI |
| ruff complexity | `max-complexity = 30` + one `# noqa: C901` | 15, no per-function escapes (harness's number) |
| docs examples | not executed | `pytest-examples`, as pydantic-ai does |
| `AGENTS.md` | unfilled template | real, modelled on harness's |
| `agent_docs/` | absent | `index`, `review-checklist`, `testing`, `core-boundary` |

## 4. Documentation

The docs are substantial (30 pages, ~6.5k lines) and structurally sound — the mkdocs nav already
mirrors backend's Concepts / Advanced / Examples / API shape. The problems are coverage gaps, stale
content, and prose that describes the library rather than teaching it.

### Undocumented public surface

| Surface | Docs mentions |
|---|---|
| `get_total_usage()` | 0 |
| `TaskHandle.traceparent`, `.trace_id`, `.span_id` | 0 |
| `TaskHandle.tool_call_counts` | 0 |
| `TaskHandle.cost`, `.model_name`, `.provider_*`, `.finish_reason` | 0 |
| `AgentFactory` | 0 |
| `usage_limits` / `UsageLimitsFactory` | 1 file |
| `send_message_to_subagent` (steering) | 1 file, no dedicated page |
| `max_result_chars` truncation | 1 file |
| `max_task_handles` | 1 file |

The whole observability story — per-task cost from genai-prices, token usage, tool-call counts, W3C
traceparent for Logfire correlation — is implemented and invisible. It is arguably the library's
strongest differentiator against a hand-rolled delegation tool.

Missing API reference pages: `registry`, `factory`, `message_bus`, `spec`, `dynamic_agent`.
`api/prompts.md` currently doubles as the retry reference.

### Stale content

- `docs/index.md` "Available Tools" table omits `wait_tasks`, `send_message_to_subagent`,
  `create_agent`, and `delegate`.
- `docs/index.md` "Core Features" lists "Pluggable Message Bus" as a headline feature; the
  pluggable half is unused by the library.
- Nesting claims and the `max_nesting_depth` annotation in `docs/examples/nesting.md` (see B11).
- `CHANGELOG.md` (309 lines) is not in the nav. Backend ships `docs/changelog.md`.

### Style target

FastAPI and pydantic-ai docs share four properties this repo's docs lack:

1. **Every page opens with a runnable snippet**, then explains it. Several pages here open with a
   concept paragraph and a bullet list.
2. **Code is tested.** pydantic-ai runs its documentation through `pytest-examples`. Adding it here
   would have caught the stale signatures in the docs directly.
3. **Progressive disclosure**, with admonitions carrying the sharp edges — `!!! warning` for the
   things that bite (a chat trace cannot be continued while its task runs; a truncation marker is a
   display limit, not an incomplete answer; `ask_user` is required for sync-mode questions).
4. **Honest "when not to use"**, which the tool descriptions in `prompts.py` already do well for
   the model. The human docs should match.

Planned additions: `concepts/observability.md`, `advanced/steering.md`, `advanced/chat-traces.md`,
`advanced/usage-limits.md`, `changelog.md`, and the five missing API pages. Planned rewrites:
`index.md` (accurate feature and tool tables), `concepts/toolset.md` (491 lines, currently the
dumping ground for every new option), `examples/nesting.md`.

Open style question: harness's `AGENTS.md` bans em-dashes and hype in all prose. These docs use both
freely. Default in this plan is to cut the hype and keep em-dashes, unless the house style should be
unified across the OSS repos.

## 5. Sequencing

Seven PRs. Each is independently reviewable and leaves `main` green. No PR breaks a downstream
import.

| PR | Scope | Acceptance |
|---|---|---|
| **1. Correctness** | B1, B5, B8, B12a–B12h | Tests assert exact model-facing strings for every tool. Regression test per defect. |
| **2. Task lifecycle** | B3, B4, B7 | `run_id` on `TaskHandle`; `wrap_run` finalizer cancels this run's tasks; cross-run lookup rejected; `ask_timeout_seconds` configurable. |
| **3. Error contract** | B2, plus `on_failure` / `contain_errors` | Control-flow signals propagate; child failure raises `ModelRetry`; deferred-tool and approval flows work inside a subagent. Documented as the one behaviour change. |
| **4. Types and tooling** | D1, D2, D3, D6 | `SubAgentToolset` is a real class; zero `type: ignore` in `src/`; pyright strict green; `make typecheck-mypy` green; ruff complexity 15 with no `noqa: C901`. Downstream smoke test against pydantic-deep. |
| **5. Core primitives** | B6, B9, B10, D5 | Steering via `AgentRun.enqueue`; `_drive_run` realigned with core plus a drift test and a pinned lower bound; upstream issue opened; `toolset.py` split behind a façade. |
| **6. Documentation** | Section 4 in full | `pytest-examples` green on every snippet; no public symbol without a docs mention; nav covers observability, steering, chat traces, usage limits, changelog, and all five missing API pages. |
| **7. Repo meta** | `AGENTS.md`, `agent_docs/`, `docs/plans/` | AGENTS.md modelled on harness's; review checklist and testing guide in place. |

PR 4 is where the compatibility budget is spent most carefully; it should land with a run of
pydantic-deep's own test suite against a local build.

## 6. Deliberately out of scope

- Breaking the API. `SubAgentConfig` stays a TypedDict rather than becoming a Pydantic model, and
  `create_subagent_toolset` keeps its 17 parameters. Both are the right long-term change and both
  would break `pydantic-deep`; they belong to a 0.4 line.
- Removing `InMemoryMessageBus.ask` / `.answer` / handlers, `create_message_bus`, and
  `MessageBusProtocol`. Retained by decision; documented as an extension surface.
- Removing `SubAgentDepsProtocol.subagents`. Deprecated in docs only.
- Run-scoped state as harness does it (`RunTasks` keyed by `run_id`, no toolset-level map). The
  downstream contract reads `task_manager.handles` globally, so PR 2 adds isolation on top of the
  shared map instead of replacing it.
