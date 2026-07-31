"""Declarative subagent specification for YAML/JSON configuration.

This module provides a Pydantic model for defining subagent configurations
in a serializable format, suitable for loading from YAML or JSON files.

Example YAML:
    ```yaml
    subagents:
      - name: researcher
        description: Research assistant
        instructions: You research topics thoroughly.
        model: openai:gpt-4.1-mini
    ```

Example usage:
    ```python
    import yaml
    from subagents_pydantic_ai.spec import SubAgentSpec

    with open("agents.yaml") as f:
        data = yaml.safe_load(f)

    specs = [SubAgentSpec(**s) for s in data["subagents"]]
    configs = [s.to_config() for s in specs]
    ```
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from subagents_pydantic_ai.types import SubAgentConfig

UNSERIALISABLE_CONFIG_KEYS = frozenset({"agent", "agent_factory", "toolsets", "retry_on"})
"""`SubAgentConfig` keys holding live Python objects, which a spec cannot carry."""

_OPTIONAL_FIELDS: tuple[str, ...] = (
    "model",
    "can_ask_questions",
    "max_questions",
    "preferred_mode",
    "typical_complexity",
    "typically_needs_context",
    "context_files",
    "agent_kwargs",
    "max_retries",
    "retry_initial_delay",
    "retry_max_delay",
    "retry_backoff_multiplier",
    "retry_jitter",
    "on_failure",
    "contain_errors",
)
"""Spec fields mapped onto `SubAgentConfig` only when explicitly set."""


class SubAgentSpec(BaseModel):
    """Declarative subagent configuration for YAML/JSON specs.

    A Pydantic model that mirrors `SubAgentConfig` (a TypedDict) but
    provides validation, defaults, and serialization support. This makes
    it suitable for loading subagent definitions from YAML or JSON files.

    Attributes:
        name: Unique identifier for the subagent.
        description: Brief description shown to the parent agent.
        instructions: System prompt for the subagent.
        model: LLM model identifier (e.g. `openai:gpt-4.1`).
            If None, the parent agent's default model is used.
        can_ask_questions: Whether the subagent can ask the parent questions.
        max_questions: Maximum number of questions per task.
        preferred_mode: Default execution mode preference.
        typical_complexity: Typical task complexity for this subagent.
        typically_needs_context: Whether this subagent typically needs user context.
        context_files: List of context file paths to inject into system prompt.
        agent_kwargs: Extra keyword arguments for the `Agent` constructor.
        max_retries: Extra attempts after a transient failure.
        retry_initial_delay: Seconds before the first retry.
        retry_max_delay: Cap for the backoff delay.
        retry_backoff_multiplier: Delay growth factor per attempt.
        retry_jitter: Randomise the backoff delay to avoid a thundering herd.
        on_failure: Message returned to the parent instead of a `ModelRetry`.
        contain_errors: Whether a crash is contained as a `ModelRetry`.
        extra: Generic extensibility dict for consumer libraries.

    The `SubAgentConfig` keys that cannot survive a round trip through YAML --
    `agent`, `agent_factory`, `toolsets`, and `retry_on`, all of which hold live
    Python objects -- have no field here. `tests/test_spec.py` fails if a new
    serialisable key is added to the config without being mirrored.
    """

    name: str
    description: str = ""
    instructions: str = ""
    model: str | None = None
    can_ask_questions: bool | None = None
    max_questions: int | None = None
    preferred_mode: Literal["sync", "async", "auto"] | None = None
    typical_complexity: Literal["simple", "moderate", "complex"] | None = None
    typically_needs_context: bool | None = None
    context_files: list[str] | None = None
    agent_kwargs: dict[str, Any] | None = None
    max_retries: int | None = Field(default=None, ge=0)
    retry_initial_delay: float | None = Field(default=None, ge=0.0)
    retry_max_delay: float | None = Field(default=None, ge=0.0)
    retry_backoff_multiplier: float | None = Field(default=None, ge=1.0)
    retry_jitter: bool | None = None
    on_failure: str | None = None
    contain_errors: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_retry_bounds(self) -> SubAgentSpec:
        """Reject a backoff whose cap is below its starting delay.

        The runtime takes `min(base, max_delay)`, so an inverted pair silently
        pins every retry to the cap instead of backing off.
        """
        if (
            self.retry_initial_delay is not None
            and self.retry_max_delay is not None
            and self.retry_max_delay < self.retry_initial_delay
        ):
            raise ValueError(
                f"retry_max_delay ({self.retry_max_delay}) must be >= "
                f"retry_initial_delay ({self.retry_initial_delay})"
            )
        return self

    def to_config(self) -> SubAgentConfig:
        """Convert to a SubAgentConfig TypedDict.

        Only fields that were explicitly set (non-`None`) are included, so a
        subagent inherits the library defaults for everything the spec left out.
        `extra` is included when non-empty.

        Returns:
            A SubAgentConfig dict suitable for `create_subagent_toolset()`.
        """
        config = SubAgentConfig(
            name=self.name,
            description=self.description,
            instructions=self.instructions,
        )

        for field_name in _OPTIONAL_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                config[field_name] = value  # type: ignore[literal-required]
        if self.extra:
            config["extra"] = self.extra

        return config

    @classmethod
    def from_config(cls, config: SubAgentConfig) -> SubAgentSpec:
        """Create a SubAgentSpec from a SubAgentConfig dict.

        Keys holding live Python objects (`agent`, `agent_factory`, `toolsets`,
        `retry_on`) are dropped: they cannot be serialised, and carrying them
        would make a spec that does not round-trip.

        Args:
            config: A SubAgentConfig TypedDict.

        Returns:
            A new SubAgentSpec instance.
        """
        data: dict[str, Any] = {
            "name": config["name"],
            "description": config.get("description", ""),
            "instructions": config.get("instructions", ""),
        }

        # A `Model` instance cannot be serialised; its string form is the best
        # available approximation.
        model_val = config.get("model")
        if model_val is not None:
            data["model"] = str(model_val)

        for field_name in (*_OPTIONAL_FIELDS, "extra"):
            if field_name != "model" and field_name in config:
                data[field_name] = config.get(field_name)

        return cls(**data)
