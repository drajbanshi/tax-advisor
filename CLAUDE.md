# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An agentic CLI chat app for tax Q&A. Python package in `src/tax_advisor/`, built with LiteLLM for multi-provider LLM support, ChromaDB for vector storage, and Presidio for PII redaction. No web service — it's a terminal REPL.

## Commands

```bash
uv sync                    # Install dependencies (use uv, not pip)
uv run tax-advisor         # Run the CLI (or: uv run python -m tax_advisor)
uv run ruff check .        # Lint
uv sync --extra bedrock    # Install with AWS Bedrock support (boto3)
```

## Architecture

**REPL loop** (`cli.py:main`) → **Agent** (`agent.py`) → **LiteLLM** (`models.py`) → streaming response with tool calls.

- `cli.py` — Interactive REPL with slash commands (`/quit`, `/clear`, `/model`, `/provider`, `/ingest`, `/upload`, `/template`, `/sessions`, etc.). Handles first-run setup (provider selection, API key prompt, IRS reference doc download).
- `agent.py` — Agentic loop: sends messages, handles streaming tool calls, enforces `MAX_TOOL_ROUNDS = 10`. Assembles streamed deltas into complete assistant messages.
- `models.py` — Thin wrapper around `litellm.completion()`. Handles Bedrock profile-based auth.
- `config.py` — `Settings` dataclass loaded from env vars (`TAX_ADVISOR_*`). Provider defaults: anthropic (Claude), openai (GPT-4o), bedrock, llama (Ollama). User config persisted to `~/.tax-advisor/.env`.
- `tools.py` — Tool definitions (OpenAI function-calling format), implementations, and registry. **When adding a tool, update all three**: `TOOL_DEFINITIONS`, implementation function, and `TOOL_REGISTRY`.

**RAG pipeline** (`rag/`):
- `pipeline.py` — Orchestrates ingest (discover → convert → redact → chunk → embed → index) and query (search both reference + session collections, merge by score, redact results).
- `index.py` — `VectorIndex` wrapping ChromaDB `PersistentClient`. Cosine similarity, deterministic chunk IDs for upsert idempotency.
- `embeddings.py` — `Embeddings` protocol with `BedrockEmbeddings` (Titan) and `OpenAIEmbeddings` backends. Concurrent batched embedding.
- `chunking.py` — Markdown-aware text splitting via `langchain-text-splitters`.
- `retrieve.py` — Query-time retrieval wrapper.

**Document handling**:
- `ingestion/loader.py` — File discovery (PDF + markdown). PDF→markdown via Docling.
- `ingestion/extract.py` — Metadata extraction from filenames/content.
- `privacy/redactor.py` — PII redaction using Microsoft Presidio (SSN, DOB, names, addresses, phone, email → `[REDACTED_*]` placeholders). Requires spaCy `en_core_web_lg` model (auto-downloaded).
- `data/__init__.py` — Bundled IRS reference docs and download logic for remote references.

**Tax form extraction** (vision-based):
- `form_classifier.py` — Classifies uploaded images as W-2, 1099, or unknown via LLM vision.
- `w2_extract.py` — Extracts W-2 fields from images via LLM vision prompts.
- `form1099_extract.py` — Extracts consolidated 1099 data (1099-B, 1099-DIV, 1099-INT) from images.
- `templates.py` — Generates YAML templates for manual W-2/1099 data entry.
- `session.py` — Session persistence (JSON files in `~/.tax-advisor/sessions/`).

**Dual-collection design**: The vector store maintains a persistent "reference" collection (IRS docs) and per-session collections (user's personal tax documents). Queries merge results from both.

## Key Conventions

- All LLM calls go through `litellm` — model strings use provider prefixes (`anthropic/`, `bedrock/`, `ollama/`).
- Messages follow OpenAI-style schema (`role`, `content`, `tool_calls`, `tool_call_id`).
- PII redaction runs at two points: pre-index (during ingestion) and pre-response (before returning RAG results to the LLM).
- User data lives in `~/.tax-advisor/` by default (override with `TAX_ADVISOR_DATA_DIR`).
- Build system is Hatch (`hatchling`). Package source is in `src/tax_advisor/`.
