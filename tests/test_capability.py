"""Tests for SubAgentCapability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.test import TestModel

from subagents_pydantic_ai import DynamicAgentRegistry, SubAgentCapability, SubAgentConfig

_MODEL = TestModel()


@dataclass
class MockDeps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> MockDeps:
        return MockDeps(subagents={} if max_depth <= 0 else self.subagents.copy())


def _cap(**kwargs):
    """Create SubAgentCapability with TestModel default."""
    kwargs.setdefault("default_model", _MODEL)
    return SubAgentCapability(**kwargs)


class TestSubAgentCapability:
    """Tests for SubAgentCapability construction and configuration."""

    def test_default_creates_toolset(self):
        """Default capability creates toolset with general-purpose subagent."""
        cap = _cap()
        assert cap.include_general_purpose is True
        assert cap.get_toolset() is not None

    def test_custom_subagents(self):
        """Custom subagent configs are accepted."""
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
            ),
        ]
        cap = _cap(subagents=configs)
        assert cap.subagents == configs
        assert cap.get_toolset() is not None

    def test_without_general_purpose(self):
        """Can disable general-purpose subagent."""
        configs = [
            SubAgentConfig(
                name="writer",
                description="Writes content",
                instructions="You are a writer.",
            ),
        ]
        cap = _cap(subagents=configs, include_general_purpose=False)
        assert cap.include_general_purpose is False

    def test_serialization_name(self):
        """Serialization name for AgentSpec."""
        assert SubAgentCapability.get_serialization_name() == "SubAgentCapability"

    def test_get_instructions_returns_callable(self):
        """get_instructions returns a callable."""
        configs = [
            SubAgentConfig(
                name="helper",
                description="Helps with tasks",
                instructions="You help.",
            ),
        ]
        cap = _cap(subagents=configs)
        instructions = cap.get_instructions()
        assert callable(instructions)

    def test_instructions_contain_subagent_names(self):
        """Dynamic instructions list available subagents."""
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="Research.",
            ),
        ]
        cap = _cap(subagents=configs)
        instructions_fn = cap.get_instructions()
        ctx = type("FakeCtx", (), {"deps": None})()
        result = instructions_fn(ctx)
        assert "researcher" in result

    def test_task_manager_property(self):
        """task_manager property is accessible."""
        cap = _cap()
        assert hasattr(cap, "task_manager")

    def test_nesting_depth(self):
        """Max nesting depth is forwarded."""
        cap = _cap(max_nesting_depth=2)
        assert cap.max_nesting_depth == 2

    def test_usage_limits_forwarded_to_toolset(self):
        """usage_limits is forwarded to the underlying toolset factory."""
        usage_limits = UsageLimits(request_limit=3, total_tokens_limit=1000)

        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap(usage_limits=usage_limits)

        assert cap.usage_limits is usage_limits
        assert create_toolset.call_args.kwargs["usage_limits"] is usage_limits

    def test_delegation_configuration_forwarded_to_toolset(self):
        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap(
                delegation_configuration="persisted_and_oneshot",
                allowed_models=["openai:gpt-4.1"],
            )

        assert cap.delegation_configuration == "persisted_and_oneshot"
        assert (
            create_toolset.call_args.kwargs["delegation_configuration"] == "persisted_and_oneshot"
        )
        assert create_toolset.call_args.kwargs["allowed_models"] == ["openai:gpt-4.1"]

    def test_delegate_tool_present_in_combined_mode(self):
        cap = _cap(
            delegation_configuration="persisted_and_oneshot",
            include_general_purpose=False,
        )
        toolset = cap.get_toolset()
        assert toolset is not None
        assert "delegate" in toolset.tools

    def test_oneshot_only_instructions_reference_delegate(self):
        cap = _cap(
            delegation_configuration="oneshot_only",
            include_general_purpose=False,
        )
        instructions_fn = cap.get_instructions()
        ctx = type("FakeCtx", (), {"deps": None})()
        result = instructions_fn(ctx)
        assert "delegate" in result
        assert "`task`" not in result

    def test_oneshot_only_rejects_configured_subagents(self):
        with pytest.raises(ValueError, match="cannot be combined with"):
            _cap(
                delegation_configuration="oneshot_only",
                subagents=[
                    SubAgentConfig(
                        name="researcher",
                        description="Researches topics",
                        instructions="Research.",
                    )
                ],
            )

    def test_oneshot_only_rejects_registry(self):
        with pytest.raises(ValueError, match="cannot be combined with a registry"):
            _cap(
                delegation_configuration="oneshot_only",
                registry=DynamicAgentRegistry(),
            )

    def test_default_mode_rejects_dynamic_agent_config(self):
        """The capability is the entry point most users configure, so it must reject too."""
        with pytest.raises(ValueError, match="exposes no dynamic-agent tool"):
            _cap(allowed_models=["openai:gpt-4.1"])

    def test_max_result_chars_defaults_to_2000(self):
        """The result preview budget defaults to 2000 characters."""
        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap()

        assert cap.max_result_chars == 2000
        assert create_toolset.call_args.kwargs["max_result_chars"] == 2000

    def test_max_result_chars_forwarded_to_toolset(self):
        """max_result_chars is forwarded to the underlying toolset factory."""
        with patch("subagents_pydantic_ai.capability.create_subagent_toolset") as create_toolset:
            cap = _cap(max_result_chars=None)

        assert cap.max_result_chars is None
        assert create_toolset.call_args.kwargs["max_result_chars"] is None


class TestSubAgentCapabilityIntegration:
    """Integration tests with real Agent."""

    @pytest.mark.anyio
    async def test_agent_with_capability(self):
        """Agent with SubAgentCapability can run successfully."""
        cap = _cap()
        agent = Agent(_MODEL, deps_type=MockDeps, capabilities=[cap])
        result = await agent.run("Delegate a task", deps=MockDeps())
        assert result.output is not None

    @pytest.mark.anyio
    async def test_agent_with_custom_subagents(self):
        """Agent with custom subagents runs successfully."""
        configs = [
            SubAgentConfig(
                name="analyst",
                description="Analyzes data",
                instructions="You analyze data.",
            ),
        ]
        cap = _cap(subagents=configs)
        agent = Agent(_MODEL, deps_type=MockDeps, capabilities=[cap])
        result = await agent.run("Analyze something", deps=MockDeps())
        assert result.output is not None

    @pytest.mark.anyio
    async def test_toolset_has_expected_tools(self):
        """Toolset has core subagent management tools."""
        cap = _cap()
        toolset = cap.get_toolset()
        assert toolset is not None
        tool_names = set(toolset.tools.keys())
        assert "task" in tool_names
        assert "check_task" in tool_names
        assert "list_active_tasks" in tool_names
