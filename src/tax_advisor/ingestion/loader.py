"""Document loading: markdown files and PDF-to-markdown conversion via Docling."""

from __future__ import annotations

from pathlib import Path

_converter = None


def _get_converter():
    """Return a cached DocumentConverter instance."""
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter

        _converter = DocumentConverter()
    return _converter


def discover_pdfs(directory: Path) -> list[Path]:
    """Recursively find all PDF files in *directory*."""
    return sorted(directory.rglob("*.pdf"))


def discover_markdowns(directory: Path) -> list[Path]:
    """Recursively find all markdown files in *directory*."""
    return sorted(directory.rglob("*.md"))


def discover_documents(directory: Path) -> list[Path]:
    """Recursively find all PDF and markdown files in *directory*."""
    return sorted([*directory.rglob("*.pdf"), *directory.rglob("*.md")])


def load_markdown(md_path: Path) -> str:
    """Read a markdown file and return its text content.

    Args:
        md_path: Path to the markdown file.

    Returns:
        The raw text content of the file.
    """
    return md_path.read_text(encoding="utf-8")


def load_pdf_as_markdown(pdf_path: Path) -> str:
    """Convert a single PDF file to a markdown string via Docling.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Markdown text extracted from the PDF.
    """
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()
