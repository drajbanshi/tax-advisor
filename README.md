# Tax Advisor

Agentic CLI for tax Q&A, built around a tool-capable chat loop.

## What this project is today
- Python CLI app (not a web service) in `src/tax_advisor/`.
- Entrypoints:
  - Console script: `tax-advisor` -> `tax_advisor.cli:main` (`pyproject.toml`)
  - Module run: `python -m tax_advisor` (`src/tax_advisor/__main__.py`)
- Current tooling includes `get_current_date` in `src/tax_advisor/tools.py`.

## Quick start
```bash
uv sync
uv run tax-advisor
```

Alternative:
```bash
uv run python -m tax_advisor
```

## Current architecture (implemented)
1. `src/tax_advisor/cli.py` (`main`) runs a REPL with slash commands (`/quit`, `/clear`, `/model <name>`).
2. User input is sent to `Agent.run(...)` in `src/tax_advisor/agent.py`.
3. `Agent` calls `chat_completion(...)` in `src/tax_advisor/models.py` (LiteLLM wrapper).
4. Streaming deltas are assembled in `Agent._handle_stream(...)` and printed as tokens arrive.
5. Tool calls are executed via `execute_tool(...)` in `src/tax_advisor/tools.py`.
6. Tool outputs are appended as `{"role": "tool", ...}` messages and the model loop continues.
7. Safety guard: `MAX_TOOL_ROUNDS = 10` prevents runaway tool recursion.

## Configuration
Environment-driven settings live in `src/tax_advisor/config.py`:
- `TAX_ADVISOR_MODEL` (default: `gpt-4o`)
- `TAX_ADVISOR_TEMPERATURE` (default: `0.3`)
- `TAX_ADVISOR_SYSTEM_PROMPT`

## Planned: document ingestion + RAG for tax documents
Goal: answer questions using uploaded tax documents while minimizing exposure of confidential data.

### Target pipeline
1. **Ingest**: load files (PDF/CSV/TXT/JSON initially) into a normalized document format.
2. **Extract**: text extraction + metadata (`source`, `doc_type`, `tax_year`, page/section).
3. **Redact (mandatory)**: detect and mask sensitive fields before chunking/indexing.
4. **Chunk + Embed**: create retrieval chunks and vector embeddings.
5. **Index**: store vectors + sanitized metadata in a vector store.
6. **Retrieve**: query-time retrieval by semantic similarity + metadata filters.
7. **Answer**: ground responses on retrieved chunks with source references.
8. **Audit**: log retrieval context and redaction events for traceability.

### Confidential data redaction policy (must-have)
Redaction should run in at least two places:
- **Pre-index redaction (required)**: no raw sensitive values are written to vector index.
- **Pre-response redaction (required)**: final model output is scanned and masked before display.

Sensitive categories to redact:
- SSN (including partial patterns)
- Date of birth (DOB)
- Physical addresses
- Person names

Recommended behavior:
- Replace with stable placeholders like `[REDACTED_SSN]`, `[REDACTED_DOB]`, `[REDACTED_NAME]`.
- Keep minimal non-sensitive context so retrieval quality remains usable.
- Fail closed: if redaction/classification fails, do not index or return raw text.

## Proposed module layout for RAG extension
- `src/tax_advisor/ingestion/loader.py` - file readers and normalization.
- `src/tax_advisor/ingestion/extract.py` - text extraction and metadata.
- `src/tax_advisor/privacy/redactor.py` - PII detection/masking policies.
- `src/tax_advisor/rag/chunking.py` - chunk creation strategy.
- `src/tax_advisor/rag/index.py` - embedding + vector index adapter.
- `src/tax_advisor/rag/retrieve.py` - retrieval orchestration.
- `src/tax_advisor/rag/pipeline.py` - end-to-end ingest/query workflows.

## Development commands
```bash
uv run ruff check .
```

## Notes
- Keep OpenAI-style message schema compatibility (`role`, `content`, `tool_calls`, `tool_call_id`).
- Tool additions should update all three places in `src/tax_advisor/tools.py`: definition, implementation, and registry.

