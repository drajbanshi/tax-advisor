"""Tool definitions and execution for the agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from tax_advisor.config import Settings

# ---------------------------------------------------------------------------
# Tool definitions in OpenAI function-calling format
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time in UTC. Useful for determining tax deadlines and filing periods.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tax_documents",
            "description": (
                "Search indexed tax documents for relevant information. "
                "This searches both IRS reference documents and any personal "
                "tax documents the user has uploaded in the current session. "
                "Use this when the user asks about specific tax rules, forms, "
                "publications, or their own tax documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "tax_year": {
                        "type": "string",
                        "description": "Optional tax year filter",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations – each function receives its parsed arguments dict
# ---------------------------------------------------------------------------


def get_current_date(**_kwargs: Any) -> str:
    """Return the current UTC date and time."""
    now = datetime.now(timezone.utc)
    return json.dumps(
        {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S UTC"),
            "iso": now.isoformat(),
        }
    )


# Lazy singleton for settings used by the search tool.
_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_tool_settings(settings: Settings) -> None:
    """Inject application settings for tools that need them (e.g. RAG)."""
    global _settings
    _settings = settings


def search_tax_documents(
    query: str,
    tax_year: str | None = None,
    n_results: int = 5,
    **_kwargs: Any,
) -> str:
    """Search indexed IRS tax documents and return matching passages."""
    from tax_advisor.rag.pipeline import query_documents

    settings = _get_settings()
    filters: dict | None = None
    if tax_year:
        filters = {"tax_year": tax_year}

    # query_documents applies PII redaction on results before returning
    context = query_documents(
        query, settings, n_results=n_results, filters=filters
    )

    return json.dumps({"query": query, "results": context})


# Map from tool name to callable
TOOL_REGISTRY: dict[str, Any] = {
    "get_current_date": get_current_date,
    "search_tax_documents": search_tax_documents,
}


def execute_tool(name: str, arguments: str) -> str:
    """Look up and execute a tool by name.

    Args:
        name: The tool function name.
        arguments: JSON-encoded arguments string.

    Returns:
        The tool result as a string.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        args = json.loads(arguments) if arguments else {}
        return func(**args)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
