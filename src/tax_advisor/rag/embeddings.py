"""Text embeddings for Bedrock Titan and OpenAI."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Protocol

from rich.console import Console

if TYPE_CHECKING:
    from tax_advisor.config import Settings

console = Console()

# Titan v2 accepts up to ~8 KB of input text per request.
_BATCH_SIZE = 20
# Number of concurrent embedding requests.
_MAX_WORKERS = 8


class Embeddings(Protocol):
    """Minimal interface that all embedding backends must satisfy."""

    def embed_query(self, text: str) -> list[float]: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class BedrockEmbeddings:
    """Generate text embeddings via Amazon Bedrock Titan.

    Args:
        model_id: The Bedrock model identifier
            (default ``amazon.titan-embed-text-v2:0``).
        profile_name: Optional AWS profile for session credentials.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        profile_name: str | None = None,
    ) -> None:
        import boto3

        self.model_id = model_id
        session_kwargs: dict[str, Any] = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._invoke(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using concurrent requests."""
        total = len(texts)
        embeddings: list[list[float] | None] = [None] * total
        done = 0
        t0 = time.monotonic()
        last_log = t0
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(self._invoke, t): idx
                for idx, t in enumerate(texts)
            }
            for future in as_completed(futures):
                embeddings[futures[future]] = future.result()
                done += 1
                now = time.monotonic()
                if done == total or now - last_log >= 2:
                    elapsed = now - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    console.print(
                        f"    Embedding {done}/{total}"
                        f"  ({rate:.1f} texts/s, ~{eta:.0f}s remaining)",
                    )
                    last_log = now
        return embeddings  # type: ignore[return-value]

    def _invoke(self, text: str) -> list[float]:
        """Call the Bedrock Titan embedding model for a single text."""
        body = json.dumps({"inputText": text})
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["embedding"]


class OpenAIEmbeddings:
    """Generate text embeddings via the OpenAI API.

    Args:
        model: The OpenAI embedding model name
            (default ``text-embedding-3-small``).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        import openai

        self.model = model
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = openai.OpenAI(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        The OpenAI API accepts multiple inputs in a single request, so we
        batch them for efficiency rather than sending one-at-a-time.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = [[] for _ in texts]
        total = len(texts)
        done = 0
        t0 = time.monotonic()
        last_log = t0

        # OpenAI supports batching natively — send in chunks of _BATCH_SIZE
        for start in range(0, total, _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            response = self._client.embeddings.create(model=self.model, input=batch)
            for item in response.data:
                all_embeddings[start + item.index] = item.embedding
            done += len(batch)
            now = time.monotonic()
            if done == total or now - last_log >= 2:
                elapsed = now - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                console.print(
                    f"    Embedding {done}/{total}"
                    f"  ({rate:.1f} texts/s, ~{eta:.0f}s remaining)",
                )
                last_log = now

        return all_embeddings


# -- Factory ------------------------------------------------------------------

_PROVIDER_DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "bedrock": "amazon.titan-embed-text-v2:0",
}


def default_embedding_model(provider: str) -> str:
    """Return the default embedding model name for *provider*."""
    return _PROVIDER_DEFAULT_EMBEDDING_MODELS.get(
        provider, "text-embedding-3-small"
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """Construct the right embeddings backend based on *settings*."""
    if settings.provider == "bedrock":
        return BedrockEmbeddings(
            model_id=settings.embedding_model,
            profile_name=settings.bedrock_profile,
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
    )
