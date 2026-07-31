# Review checklist

## Product fit

- The behaviour belongs here, not in Pydantic AI core (`core-boundary.md`).
- The public surface is small and named around user concepts.
- An existing core primitive was checked for first.

## Implementation

- Public exports are intentional; private helpers stay private.
- No `Any` in a new public signature, and no `type: ignore` in `src/`.
- Every invariant in `delegation-contracts.md` still holds.
- Model-facing strings render enums as values.
- Anything accumulating per delegation is bounded.
- Any new `task_id` tool goes through `_handle_for`.
- Dependency changes went through `uv` and have a reason.

## Tests

- New behaviour has tests; a fixed bug has a regression test naming it.
- Tests assert the output a caller or a model sees.
- `make lint && make typecheck-both && make test` pass, with coverage at 100%.

## Compatibility

- pydantic-deep still passes:
  `cd ../pydantic-deep && PYTHONPATH=../pydantic-ai-subagents/src uv run pytest -q`
- No public name, signature, or module path moved without a shim.

## Docs

- `docs/` covers the change, and `uv run mkdocs build --strict` is clean.
- `tests/test_docs.py` passes.
- `CHANGELOG.md` has an entry for anything user-visible, saying what broke and why
  it mattered -- not just what changed.
