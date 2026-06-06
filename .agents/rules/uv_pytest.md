---
trigger: always_on
description: "Always run Python commands and tests with `uv run` in this project."
---

# Use `uv run` for Python Commands and Tests

In this project, always run Python commands, scripts, linters, type checkers, and tests using the `uv run` command.

## Instructions:
1. When running tests, always use `uv run pytest` instead of `pytest`.
2. When running the linter, always use `uv run ruff` instead of `ruff`.
3. When running the type checker, always use `uv run ty` instead of `ty`.
4. When executing any Python scripts, use `uv run python <script_path>`.
