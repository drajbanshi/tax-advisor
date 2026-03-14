"""ChromaDB vector index for document chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from tax_advisor.rag.chunking import Chunk
from tax_advisor.rag.embeddings import Embeddings


@dataclass
class SearchResult:
    """A single search result from the vector index."""

    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class VectorIndex:
    """Persistent vector store backed by ChromaDB.

    Args:
        chroma_dir: Directory for ChromaDB persistent storage.
        collection_name: Name of the ChromaDB collection.
        embeddings: An embeddings backend satisfying the :class:`Embeddings` protocol.
    """

    def __init__(
        self,
        chroma_dir: str | Path,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        self._embeddings = embeddings
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert *chunks* into the collection.

        Each chunk is assigned a deterministic ID based on its metadata
        ``source`` and position so that re-ingestion overwrites rather than
        duplicates.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._embeddings.embed_texts(texts)

        ids: list[str] = []
        metadatas: list[dict] = []
        for idx, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            chunk_id = f"{source}::chunk_{idx}"
            ids.append(chunk_id)
            metadatas.append(chunk.metadata)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search for chunks similar to *query*.

        Args:
            query: Natural-language search query.
            n_results: Maximum number of results to return.
            filters: Optional ChromaDB metadata ``where`` filter dict.

        Returns:
            Ranked list of :class:`SearchResult` objects (best first).
        """
        query_embedding = self._embeddings.embed_query(query)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to a similarity score in [0, 1].
            score = 1.0 - (dist / 2.0)
            search_results.append(
                SearchResult(text=doc, metadata=meta or {}, score=score)
            )

        return search_results

    def delete_collection(self) -> None:
        """Delete the underlying ChromaDB collection."""
        self._client.delete_collection(name=self._collection.name)

    def stats(self) -> dict:
        """Return basic statistics about the collection."""
        count = self._collection.count()
        return {
            "collection": self._collection.name,
            "document_count": count,
        }
