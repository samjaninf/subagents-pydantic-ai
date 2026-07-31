# Message bus API

The bus delivers parent-to-child steering messages for background delegations. See
[Steering](../advanced/steering.md) and [Message bus](../advanced/message-bus.md).

`TaskManager` owns the lifecycle of every background delegation: its
`asyncio.Task`, its `TaskHandle`, its cancellation event, and the future a blocked
`ask_parent` waits on.

## TaskManager

::: subagents_pydantic_ai.TaskManager
    options:
      show_root_heading: true
      show_source: true

## InMemoryMessageBus

::: subagents_pydantic_ai.InMemoryMessageBus
    options:
      show_root_heading: true
      show_source: true

## create_message_bus

::: subagents_pydantic_ai.create_message_bus
    options:
      show_root_heading: true
      show_source: true
