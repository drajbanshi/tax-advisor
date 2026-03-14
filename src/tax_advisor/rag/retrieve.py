"""Retrieval orchestration."""

from __future__ import annotations

from tax_advisor.rag.index import SearchResult, VectorIndex


def retrieve(
    query: str,
    index: VectorIndex,
    n_results: int = 5,
    filters: dict | None = None,
) -> list[SearchResult]:
    """Retrieve the most relevant chunks for *query*.

    Args:
        query: Natural-language search query.
        index: The :class:`VectorIndex` to search.
        n_results: Maximum number of results.
        filters: Optional metadata filters forwarded to ChromaDB.

    Returns:
        Ranked list of :class:`SearchResult` objects.
    """
    return index.search(query, n_results=n_results, filters=filters)
