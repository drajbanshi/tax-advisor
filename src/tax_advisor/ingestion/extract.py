"""Metadata extraction from document paths and markdown document wrapper."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MarkdownDocument:
    """Holds converted markdown text together with its metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


def extract_metadata(file_path: Path, *, tax_year: str | None = None) -> dict:
    """Extract metadata from a document filename and path.

    Supports both PDF and markdown files.

    Heuristics:
    - ``source``: the filename without extension.
    - ``tax_year``: uses the explicit *tax_year* when provided, otherwise
      attempts to parse a four-digit year from the filename.
    - ``doc_type``: "publication" if the name contains "pub" or starts with
      ``p`` followed by digits, "form" if it contains "form" or starts with
      ``f`` followed by digits, otherwise "other".

    Args:
        file_path: Path to the document file (PDF or markdown).
        tax_year: Optional explicit tax year (e.g. ``"2025"``).  When given,
            this value is used directly instead of trying to parse a year from
            the filename.

    Returns:
        A dict of extracted metadata fields.
    """
    stem = file_path.stem.lower()
    metadata: dict = {"source": file_path.name}

    # Tax year: prefer explicit value, fall back to filename heuristic
    if tax_year is not None:
        metadata["tax_year"] = tax_year
    else:
        year_match = re.search(r"(20\d{2})", stem)
        if year_match:
            metadata["tax_year"] = year_match.group(1)

    # Classify document type
    if "pub" in stem or re.match(r"^p\d+", stem):
        metadata["doc_type"] = "publication"
    elif "form" in stem or re.match(r"^f\d+", stem):
        metadata["doc_type"] = "form"
    else:
        metadata["doc_type"] = "other"

    return metadata
