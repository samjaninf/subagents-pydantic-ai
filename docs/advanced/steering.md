# Steering a running subagent

A background subagent can be redirected mid-flight. The parent sends a message,
the subagent folds it into its next model request, and everything it has done so
far survives.

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        SubAgentCapability(
            subagents=[
                SubAgentConfig(
                    name="searcher",
                    description="Searches a codebase",
                    instructions="You search codebases and report matches.",
                ),
            ],
        )
    ],
)

result = await agent.run(
    "Search the repo for the retry logic in the background. "
    "If you learn it lives in packages/sparta, tell the searcher to narrow down."
)
```

The model does this with two tools:

```text
task(description="Find the retry logic", subagent_type="searcher", mode="async")
-> Task ID: a1b2c3d4

send_message_to_subagent(task_id="a1b2c3d4", message="Narrow to packages/sparta/, it isn't in core/")
-> Message delivered to task 'a1b2c3d4'; it will be applied on the subagent's next step.
```

## Why not cancel and re-delegate

Cancelling throws away every tool call the subagent already made and every
conclusion it already reached, then pays for all of it again. Steering keeps the
run alive: the message arrives as an extra user instruction on the next model
turn, alongside the history the subagent has built up.

Reach for steering when you learn something the subagent needs:

- narrowing a search that is going too wide
- an early stop ("the first five matches are enough")
- a correction ("that module was renamed last month")
- a change of priority ("skip the tests, I only need the public API")

## Steering versus answering

Two tools deliver text to a running subagent, and they are not interchangeable.

| | `send_message_to_subagent` | `answer_subagent` |
|---|---|---|
| Who starts it | The parent, unprompted | The subagent, by calling `ask_parent` |
| Task status | `running` | `waiting_for_answer` |
| Effect | Extra instruction on the next model request | Resolves the call the subagent is blocked on |
| If it never arrives | The subagent carries on as before | The subagent blocks until `ask_timeout_seconds` |

See [Questions](questions.md) for the `ask_parent` side.

## How delivery works

Steering rides pydantic-ai's own `AgentRun.enqueue`, so the library is not
splicing message parts by hand:

1. `send_message_to_subagent` puts the message on the internal bus, keyed by the
   task id.
2. The subagent's driving loop drains the bus between graph nodes.
3. Anything pending goes to `enqueue`, and core delivers it before the next model
   request.

Because core owns placement, a steering message can never be spliced between a
tool call and its return -- a shape most providers reject outright.

!!! warning "Background tasks only"
    Steering needs a live run to deliver into. In sync mode the parent's own run
    loop is blocked inside the delegation, so there is no turn in which it could
    send anything; use `ask_user` and [Questions](questions.md) instead. A task
    that has already finished returns an error naming its status.

## Isolation

A steering message reaches a task only if the calling run started it:

```python
toolset = create_subagent_toolset()  # one instance, shared by every run
```

One toolset instance typically serves every run of its agent, so tasks record the
`run_id` that started them and the tools refuse ids belonging to another run. In a
server sharing one agent across users, run B cannot steer -- or read, answer, or
cancel -- run A's task by guessing its id.

## Ordering

Messages are delivered in the order they were sent, and several sent before the
subagent's next step arrive together on that step. Nothing is dropped while the
task is running, and nothing is queued for a task that has finished.

## Next steps

- [Questions](questions.md) -- the subagent asking the parent
- [Cancellation](cancellation.md) -- when steering is not enough
- [Execution modes](execution-modes.md) -- why steering needs background mode
