"""Section-aware markdown chunking using langchain-text-splitters."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


@dataclass
class Chunk:
    """A single text chunk with associated metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


# Approximate characters-per-token ratio for English text.
_CHARS_PER_TOKEN = 4
_MAX_CHUNK_TOKENS = 1000
_MAX_CHUNK_CHARS = _MAX_CHUNK_TOKENS * _CHARS_PER_TOKEN
_OVERLAP_CHARS = 200

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_CHUNK_CHARS,
    chunk_overlap=_OVERLAP_CHARS,
)


def chunk_markdown(markdown: str, metadata: dict | None = None) -> list[Chunk]:
    """Split *markdown* into section-aware chunks.

    Uses a two-stage approach:
    1. ``MarkdownHeaderTextSplitter`` splits on headings and captures
       the header hierarchy as metadata.
    2. ``RecursiveCharacterTextSplitter`` enforces maximum chunk size
       with character overlap for continuity.

    Args:
        markdown: Full markdown text to chunk.
        metadata: Base metadata dict to attach to every chunk.

    Returns:
        List of :class:`Chunk` objects.
    """
    base_meta = metadata or {}

    # Stage 1: split by markdown headers
    header_docs = _header_splitter.split_text(markdown)

    # Stage 2: enforce chunk size limits
    split_docs = _text_splitter.split_documents(header_docs)

    # Map langchain Documents → Chunk dataclass
    chunks: list[Chunk] = []
    for doc in split_docs:
        chunk_meta = {**base_meta}
        # Derive section_title from deepest header level present
        for key in ("h3", "h2", "h1"):
            if key in doc.metadata:
                chunk_meta["section_title"] = doc.metadata[key]
                break
        chunks.append(Chunk(text=doc.page_content, metadata=chunk_meta))

    return chunks
