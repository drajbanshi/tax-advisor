# AGENTS.md

## Purpose and Scope
- This repo is a small agentic CLI for tax Q&A (`src/tax_advisor/`), not a web service.
- Primary entrypoint is the console script `tax-advisor` (`pyproject.toml` -> `tax_advisor.cli:main`).
- `README.md` is currently empty; treat code and `pyproject.toml` as the source of truth.

## Architecture You Need First
- User input loop lives in `src/tax_advisor/cli.py` (`main`). It handles `/quit`, `/clear`, `/model <name>` before invoking the agent.
- Conversation state is owned by `src/tax_advisor/agent.py` (`Agent.messages`), seeded with a single `system` prompt from `Settings`.
- The core execution path is:
  1. CLI collects text -> `Agent.run(text)`
  2. `Agent` calls `chat_completion(...)` in `src/tax_advisor/models.py`
  3. Streaming deltas are assembled in `_handle_stream`
  4. Any tool calls are executed via `execute_tool(...)` in `src/tax_advisor/tools.py`
  5. Tool results are appended as `{"role": "tool", ...}` messages and the loop continues
- Tool loop is intentionally capped by `MAX_TOOL_ROUNDS = 10` to avoid infinite tool recursion.

## Model and Config Conventions
- Settings are env-driven dataclass fields in `src/tax_advisor/config.py`:
  - `TAX_ADVISOR_MODEL` (default `gpt-4o`)
  - `TAX_ADVISOR_TEMPERATURE` (default `0.3`)
  - `TAX_ADVISOR_SYSTEM_PROMPT` (long default tax-assistant prompt)
- Model provider abstraction is intentionally thin: `chat_completion(...)` passes OpenAI-style message/tool payloads directly to `litellm.completion(...)`.
- Keep message dictionaries OpenAI-compatible (`role`, `content`, optional `tool_calls`, `tool_call_id`).

## Tooling Pattern (Important)
- Add tools in 3 synchronized places in `src/tax_advisor/tools.py`:
  1. `TOOL_DEFINITIONS` (LLM-visible JSON schema)
  2. Python implementation function (returns a string, usually JSON)
  3. `TOOL_REGISTRY` mapping
- `execute_tool(name, arguments)` expects `arguments` as a JSON string; it handles empty args and returns JSON error strings for unknown/failing tools.
- Existing example to mirror: `get_current_date` (no required params, returns `date/time/iso` JSON).

## CLI/UX Behavior to Preserve
- Keep token streaming behavior in `Agent._handle_stream` (prints partial text as chunks arrive).
- Keep Rich-based console formatting (`Panel`, markdown rendering in non-streaming mode) and prompt-toolkit REPL behavior.
- `/model` mutates `settings.model` in-place; ongoing conversation history is preserved unless `/clear` is used.

## Developer Workflows (Observed + Inferred)
- Python requirement is `>=3.11` (`pyproject.toml`), with local `.python-version` set to `3.13`.
- Dependency management appears to use `uv` (`uv.lock` present).
- Common commands (from project metadata):
  - `uv sync`
  - `uv run tax-advisor`
  - `uv run python -m tax_advisor`
  - `uv run ruff check .`
- There are currently no test files in the repository; if adding behavior, add focused tests around `tools.py` parsing/execution and `agent.py` tool-loop transitions first.

