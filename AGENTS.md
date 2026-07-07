# AGENTS.md

Shared instructions for AI coding agents working in this repository. This is the
cross-tool source of truth (read by Codex, Cursor, Gemini CLI, Antigravity,
OpenCode, and others). Claude Code loads it via `@AGENTS.md` in `CLAUDE.md`.

## Python: always use `uv run`

Run all Python commands, scripts, linters, type checkers, and tests through `uv run`:

- Tests: `uv run pytest` (never bare `pytest`)
- Linter: `uv run ruff` (never bare `ruff`)
- Type checker: `uv run ty` (never bare `ty`)
- Scripts: `uv run python <script_path>`
