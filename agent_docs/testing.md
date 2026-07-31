# Testing

## Rules

- No live model calls, ever. `pydantic_ai.models.test.TestModel` for anything
  needing a real `Agent`; `StubAgent` (`tests/test_lifecycle.py`) or `FakeAgent`
  (`tests/test_toolset.py`) when you want to control the run.
- The fakes drive `.iter()` the way `Agent.run` does, so they exercise the real
  execution path rather than a shortcut around it.
- `asyncio_mode = "auto"`: async tests need no marker, and adding
  `@pytest.mark.asyncio` to a sync test raises a warning.
- Construct the toolset with `default_model=TestModel()`. The `"openai:gpt-4.1"`
  default needs the `openai` package at construction time.

## What a good test asserts

Assert what a caller or a model receives.

```python
# Weak: the branch ran.
assert handle.status == TaskStatus.COMPLETED

# Strong: the model was told the truth.
result = await toolset.tools["check_task"].function(ctx, "t1")
assert "Status: completed" in result
assert "TaskStatus." not in result
```

100% branch coverage is the gate, and it is not enough on its own -- the suite
reported 100% while every `check_task` call leaked an enum member name.

## Regression tests

A test that pins a defect says so in its docstring: what broke, and why it was not
caught. `tests/test_lifecycle.py` is the home for cross-module ones (run isolation,
cancellation, terminal transitions, status rendering).

## Background tasks

`asyncio.create_task` schedules a task; it has not started when the call returns.
`await asyncio.sleep(0)` before asserting on a running task, or before cancelling
one you want to observe stopping mid-run.

Release blocked tasks at the end of a test:

```python
block.set()
await asyncio.gather(*toolset.task_manager.tasks.values(), return_exceptions=True)
```

## Docs

`tests/test_docs.py` checks that every fenced `python` block in `docs/` and
`README.md` parses, imports names that exist, and passes real keyword arguments to
the documented entry points. Snippets are not executed.

Illustrative blocks -- diagrams, `...` elisions, prose comparisons -- belong in a
```text fence, not ```python.
