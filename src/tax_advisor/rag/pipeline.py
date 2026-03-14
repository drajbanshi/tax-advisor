"""End-to-end ingestion and query pipelines."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from tax_advisor.config import Settings
from tax_advisor.ingestion.extract import MarkdownDocument, extract_metadata
from tax_advisor.ingestion.loader import (
    discover_documents,
    load_markdown,
    load_pdf_as_markdown,
)
from tax_advisor.rag.chunking import chunk_markdown
from tax_advisor.rag.embeddings import build_embeddings
from tax_advisor.rag.index import SearchResult, VectorIndex
from tax_advisor.rag.retrieve import retrieve


def _build_index(settings: Settings) -> VectorIndex:
    """Construct a :class:`VectorIndex` from application settings."""
    embeddings = build_embeddings(settings)
    return VectorIndex(
        chroma_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=embeddings,
    )


def _build_session_index(settings: Settings) -> VectorIndex | None:
    """Build a :class:`VectorIndex` for the active session collection.

    Returns ``None`` when no session is active.
    """
    if not settings.session_collection:
        return None
    embeddings = build_embeddings(settings)
    return VectorIndex(
        chroma_dir=settings.chroma_dir,
        collection_name=settings.session_collection,
        embeddings=embeddings,
    )


def ingest_documents(
    docs_dir: Path,
    settings: Settings,
    console: Console,
    *,
    skip_redaction: bool = False,
    tax_year: str | None = None,
    collection_override: str | None = None,
) -> int:
    """Discover, convert, redact, chunk, embed, and index documents.

    Supports both PDF and markdown (``.md``) files.  Markdown files are
    loaded directly; PDFs are converted to markdown via Docling.

    Args:
        docs_dir: Directory containing PDF and/or markdown files.
        settings: Application settings (embedding model, chroma config, …).
        console: Rich console for progress output.
        skip_redaction: If ``True``, skip PII redaction (useful for public
            IRS reference documents where redaction would corrupt content).
        tax_year: Optional tax year to attach to all ingested documents.
            When provided, overrides year detection from filenames.
        collection_override: When set, ingest into this collection name
            instead of the default persistent collection.

    Returns:
        The number of documents successfully ingested.
    """
    docs = discover_documents(docs_dir)
    if not docs:
        console.print(f"[yellow]No PDF or markdown files found in {docs_dir}[/yellow]")
        return 0

    console.print(f"Found [bold]{len(docs)}[/bold] document(s) in {docs_dir}")

    redactor = None
    if not skip_redaction:
        from tax_advisor.privacy.redactor import Redactor

        redactor = Redactor()

    if collection_override:
        embeddings = build_embeddings(settings)
        index = VectorIndex(
            chroma_dir=settings.chroma_dir,
            collection_name=collection_override,
            embeddings=embeddings,
        )
    else:
        index = _build_index(settings)

    ingested = 0

    for i, doc_path in enumerate(docs, 1):
        size_mb = doc_path.stat().st_size / (1024 * 1024)
        console.print(
            f"  [{i}/{len(docs)}] [cyan]{doc_path.name}[/cyan] ({size_mb:.1f} MB)"
        )

        try:
            # Load document content as markdown
            if doc_path.suffix.lower() == ".md":
                console.print("    Loading markdown …")
                t0 = time.monotonic()
                markdown = load_markdown(doc_path)
                elapsed = time.monotonic() - t0
                console.print(f"    Loaded in {elapsed:.1f}s")
            else:
                console.print("    Converting PDF to markdown …")
                t0 = time.monotonic()
                markdown = load_pdf_as_markdown(doc_path)
                elapsed = time.monotonic() - t0
                console.print(f"    Converted in {elapsed:.1f}s")

            metadata = extract_metadata(doc_path, tax_year=tax_year)
            doc = MarkdownDocument(text=markdown, metadata=metadata)

            # Redact PII before chunking/indexing
            if redactor is not None:
                console.print("    Redacting PII …")
                doc.text = redactor.redact(doc.text)

            # Chunk
            console.print("    Chunking …")
            chunks = chunk_markdown(doc.text, metadata=doc.metadata)
            console.print(f"    → {len(chunks)} chunk(s)")

            # Embed & index
            console.print("    Embedding & indexing …")
            index.add_chunks(chunks)
            ingested += 1
            console.print(f"    [green]✓ indexed[/green]")
        except Exception as exc:
            console.print(f"    [red]✗ error: {exc}[/red]")

    console.print(
        f"\n[bold green]Ingestion complete:[/bold green] "
        f"{ingested}/{len(docs)} document(s) indexed."
    )
    return ingested


def ingest_markdown_text(
    text: str,
    source_name: str,
    settings: Settings,
    console: Console,
    *,
    collection_override: str | None = None,
    skip_redaction: bool = False,
    tax_year: str | None = None,
) -> int:
    """Redact, chunk, embed, and index an in-memory markdown string.

    This follows the same pipeline as :func:`ingest_documents` but operates
    on a markdown string rather than discovering files on disk.

    Args:
        text: Markdown text to ingest.
        source_name: A logical source name for metadata (e.g. ``"w2-upload"``).
        settings: Application settings.
        console: Rich console for progress output.
        collection_override: Target collection name.  Falls back to the
            default persistent collection when ``None``.
        skip_redaction: If ``True``, skip PII redaction.
        tax_year: Optional tax year to attach as metadata.

    Returns:
        ``1`` on success, ``0`` on failure.
    """
    redactor = None
    if not skip_redaction:
        from tax_advisor.privacy.redactor import Redactor

        redactor = Redactor()

    if collection_override:
        embeddings = build_embeddings(settings)
        index = VectorIndex(
            chroma_dir=settings.chroma_dir,
            collection_name=collection_override,
            embeddings=embeddings,
        )
    else:
        index = _build_index(settings)

    try:
        # Build metadata
        metadata: dict[str, str] = {"source": source_name, "doc_type": "form"}
        if tax_year:
            metadata["tax_year"] = tax_year

        # Redact PII before chunking/indexing
        content = text
        if redactor is not None:
            console.print("    Redacting PII …")
            content = redactor.redact(content)

        # Chunk
        console.print("    Chunking …")
        chunks = chunk_markdown(content, metadata=metadata)
        console.print(f"    → {len(chunks)} chunk(s)")

        # Embed & index
        console.print("    Embedding & indexing …")
        index.add_chunks(chunks)
        console.print("    [green]✓ indexed[/green]")
        return 1
    except Exception as exc:
        console.print(f"    [red]✗ error: {exc}[/red]")
        return 0


def query_documents(
    query: str,
    settings: Settings,
    n_results: int = 5,
    filters: dict | None = None,
) -> str:
    """Retrieve relevant chunks and format them as context for the LLM.

    Queries both the persistent IRS reference collection and the active
    session collection (if any), merges results by score, and returns the
    top *n_results*.

    Args:
        query: The user's search query.
        settings: Application settings.
        n_results: Maximum number of chunks to return.
        filters: Optional metadata filters (e.g. ``{"tax_year": "2024"}``).

    Returns:
        A formatted string containing the retrieved context passages.
    """
    # Query the persistent reference collection
    index = _build_index(settings)
    results: list[SearchResult] = retrieve(
        query, index, n_results=n_results, filters=filters,
    )

    # Query the session collection (if active) and merge
    session_index = _build_session_index(settings)
    if session_index is not None:
        session_results = retrieve(
            query, session_index, n_results=n_results, filters=filters,
        )
        results = sorted(
            results + session_results, key=lambda r: r.score, reverse=True,
        )[:n_results]

    if not results:
        return "No relevant documents found."

    # Redact PII in retrieved text before returning to the LLM
    from tax_advisor.privacy.redactor import Redactor

    redactor = Redactor()

    parts: list[str] = []
    for i, result in enumerate(results, 1):
        source = result.metadata.get("source", "unknown")
        section = result.metadata.get("section_title", "")
        year = result.metadata.get("tax_year", "")
        header = f"[{i}] {source}"
        if year:
            header += f" (tax year {year})"
        if section:
            header += f" — {section}"
        text = redactor.redact(result.text)
        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def get_index_stats(settings: Settings) -> dict:
    """Return statistics about the current vector index.

    Args:
        settings: Application settings.

    Returns:
        A dict with ``reference`` stats and optionally ``session`` stats.
    """
    index = _build_index(settings)
    stats: dict = {"reference": index.stats()}

    session_index = _build_session_index(settings)
    if session_index is not None:
        stats["session"] = session_index.stats()

    return stats
