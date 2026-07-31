# Dynamic agent API

Helpers shared by the `create_agent` and `delegate` tools and by
`create_agent_factory_toolset`: they validate a runtime agent request and build the
agent instance. See [Dynamic agents](../advanced/dynamic-agents.md).

## AgentFactory

::: subagents_pydantic_ai.AgentFactory
    options:
      show_root_heading: true

## build_dynamic_agent

::: subagents_pydantic_ai.dynamic_agent.build_dynamic_agent
    options:
      show_root_heading: true
      show_source: true

## Validation helpers

::: subagents_pydantic_ai.dynamic_agent
    options:
      show_root_heading: true
      show_source: true
      members:
        - validate_agent_name
        - validate_model
        - validate_capabilities
        - validate_capabilities_with_factory
        - build_subagent_config
        - collect_agent_toolsets
        - build_agent_instance
