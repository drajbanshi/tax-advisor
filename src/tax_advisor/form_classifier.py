"""Classify tax form images (W-2 vs 1099) using LLM vision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tax_advisor.models import chat_completion
from tax_advisor.w2_extract import load_image_as_base64

_CLASSIFICATION_PROMPT = """\
You are a tax document classifier. Look at this image and determine what type \
of tax form it is. Respond with exactly one of these labels:

- w2 — if this is a W-2 (Wage and Tax Statement)
- 1099 — if this is any type of 1099 form (1099-B, 1099-DIV, 1099-INT, \
1099-MISC, 1099-NEC, consolidated 1099, etc.)
- unknown — if you cannot determine the form type

Return ONLY the label, nothing else.
"""


def classify_form(
    image_path: str | Path,
    model: str,
    bedrock_profile: str | None = None,
) -> str:
    """Classify a tax form image as ``"w2"``, ``"1099"``, or ``"unknown"``.

    Args:
        image_path: Path to the tax form image.
        model: litellm model identifier.
        bedrock_profile: Optional AWS profile for Bedrock models.

    Returns:
        One of ``"w2"``, ``"1099"``, or ``"unknown"``.
    """
    b64, mime = load_image_as_base64(image_path)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _CLASSIFICATION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                },
            ],
        }
    ]

    response = chat_completion(
        messages=messages,
        model=model,
        temperature=0.0,
        stream=False,
        bedrock_profile=bedrock_profile,
    )

    raw = response.choices[0].message.content.strip().lower()

    # Normalize the response to one of the expected labels
    if "w2" in raw or "w-2" in raw:
        return "w2"
    if "1099" in raw:
        return "1099"
    return "unknown"
