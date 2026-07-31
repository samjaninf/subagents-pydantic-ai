# Core boundary

This library extends Pydantic AI through public primitives. It should not become a
second runtime.

## Belongs in Pydantic AI core

- new agent loop semantics
- new normalized message parts, or message-history behaviour
- provider or model capability facts
- generic tool execution semantics
- durable execution primitives
- new capability hooks, or changes to hook ordering

## Belongs here

- which specialists exist and how the model is told about them
- how a task is dispatched: sync, background, or chosen from characteristics
- what a delegation failure looks like to the parent model
- steering, cancellation, and question policy
- what telemetry a delegation records
- memory bounds on retained tasks and conversations

## Decision rule

If a feature would be hard for any third-party toolset to implement through public
Pydantic AI APIs, do not work around it here. Identify the missing primitive and
propose that change first.

`retry.py` is the standing exception, and it is not a comfortable one. Driving a run
with a custom step while keeping event streaming has no public equivalent, so
`_drive_run` mirrors the loop inside `Agent.run` using private names. That copy has
drifted from core once already. Its module docstring records the coupling; the fix
is a public primitive in core, not more copying.

## Primitives worth checking before writing your own

| Need | Core primitive |
|---|---|
| Inject a message into a live run | `AgentRun.enqueue` / `RunContext.enqueue` |
| Identify the current run | `RunContext.run_id` |
| Drive a run node by node with hooks intact | `AgentRun.next` |
| Recover a failed attempt's messages | `AgentRun.all_messages()` |
| Wrap a whole run | `AbstractCapability.wrap_run` |
| Cap tokens or requests | `UsageLimits` |
