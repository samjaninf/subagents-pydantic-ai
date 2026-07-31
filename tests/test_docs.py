"""Checks that the documentation's Python snippets still match the library.

Docs rot silently: a renamed parameter or a moved export leaves the prose looking
fine while every reader who copies the snippet hits a `TypeError`. These tests
catch the two failure modes that actually happen.

1. **Syntax** -- every fenced `python` block in `docs/` and `README.md` compiles.
2. **API drift** -- every name a snippet imports from `subagents_pydantic_ai`
   exists, and every keyword argument it passes to a documented entry point is
   really a parameter of that function.

Snippets are not executed: most need a model, an event loop, and credentials.
Execution is what the rest of the suite is for.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import subagents_pydantic_ai
from subagents_pydantic_ai import (
    SubAgentCapability,
    create_agent_factory_toolset,
    create_subagent_toolset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

_FENCE = re.compile(r"^```(?:py|python)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)

#: Entry points whose keyword arguments the snippets are checked against.
_CHECKED_CALLABLES: dict[str, Any] = {
    "create_subagent_toolset": create_subagent_toolset,
    "create_agent_factory_toolset": create_agent_factory_toolset,
    "SubAgentCapability": SubAgentCapability,
}


@dataclass(frozen=True)
class Snippet:
    """One fenced Python block, with enough context to name it in a failure."""

    path: Path
    index: int
    source: str

    @property
    def id(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}#{self.index}"


def _markdown_files() -> list[Path]:
    files = sorted(DOCS_DIR.rglob("*.md"))
    files.append(REPO_ROOT / "README.md")
    return [path for path in files if "plans" not in path.parts]


def _snippets() -> list[Snippet]:
    found: list[Snippet] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(_FENCE.finditer(text)):
            found.append(Snippet(path=path, index=index, source=match.group(1)))
    return found


SNIPPETS = _snippets()


def test_snippets_were_found() -> None:
    """Guard against a regex that silently stops matching."""
    assert len(SNIPPETS) > 40


@pytest.mark.parametrize("snippet", SNIPPETS, ids=lambda s: s.id)
def test_snippet_compiles(snippet: Snippet) -> None:
    """A documented snippet is at least valid Python."""
    try:
        ast.parse(snippet.source)
    except SyntaxError as exc:  # pragma: no cover - only on a broken snippet
        pytest.fail(f"{snippet.id} does not parse: {exc}")


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "subagents_pydantic_ai"
        ):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("snippet", SNIPPETS, ids=lambda s: s.id)
def test_snippet_imports_exist(snippet: Snippet) -> None:
    """Every name a snippet imports from the library is really exported."""
    tree = ast.parse(snippet.source)
    missing = sorted(
        name for name in _imported_names(tree) if not hasattr(subagents_pydantic_ai, name)
    )
    assert not missing, f"{snippet.id} imports names the library does not export: {missing}"


def _keyword_calls(tree: ast.Module) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        if name is None or name not in _CHECKED_CALLABLES:
            continue
        calls.append((name, [kw.arg for kw in node.keywords if kw.arg is not None]))
    return calls


@pytest.mark.parametrize("snippet", SNIPPETS, ids=lambda s: s.id)
def test_snippet_keyword_arguments_exist(snippet: Snippet) -> None:
    """Every keyword a snippet passes to an entry point is a real parameter."""
    tree = ast.parse(snippet.source)
    for name, keywords in _keyword_calls(tree):
        parameters = inspect.signature(_CHECKED_CALLABLES[name]).parameters
        unknown = sorted(kw for kw in keywords if kw not in parameters)
        assert not unknown, f"{snippet.id} passes unknown arguments to {name}(): {unknown}"
