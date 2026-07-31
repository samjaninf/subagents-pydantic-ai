"""Test package.

Making `tests` a package gives its modules fully-qualified names
(`tests.test_toolset`), which is what mypy's `module = "tests.*"` override needs
to match. Without it the override silently applied to nothing.
"""
