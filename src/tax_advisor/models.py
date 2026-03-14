"""Thin wrapper around litellm for model interaction."""

from __future__ import annotations

import os
from typing import Any

import litellm

# Suppress litellm's noisy logging by default.
litellm.suppress_debug_info = True


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float = 0.3,
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
    bedrock_profile: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Send a chat completion request via litellm.

    Args:
        messages: Conversation history in OpenAI message format.
        model: Model string (e.g. "gpt-4o", "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0").
        temperature: Sampling temperature.
        tools: Optional tool definitions in OpenAI function-calling format.
        stream: Whether to stream the response.
        bedrock_profile: AWS profile used when calling Bedrock-backed models.
        api_key: Optional API key passed through to litellm (e.g. OpenAI key).

    Returns:
        A ModelResponse or a generator of streaming chunks.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        kwargs["tools"] = tools
    if api_key:
        kwargs["api_key"] = api_key

    if model.lower().startswith("bedrock/"):
        try:
            import boto3  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Bedrock provider requires boto3. Run `uv sync` to install dependencies."
            ) from exc

    if model.lower().startswith("bedrock/") and bedrock_profile:
        # Prefer profile-based auth for Bedrock.
        kwargs["aws_profile_name"] = bedrock_profile
        os.environ["AWS_PROFILE"] = bedrock_profile

    return litellm.completion(**kwargs)
