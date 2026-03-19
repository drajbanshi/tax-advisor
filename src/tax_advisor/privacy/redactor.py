"""PII redaction using Microsoft Presidio."""

from __future__ import annotations

import shutil
import subprocess
import sys

import spacy.about
import spacy.util
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Map Presidio entity types to redaction placeholders.
_ENTITY_OPERATOR_MAP: dict[str, str] = {
    "US_SSN": "[REDACTED_SSN]",
    "DATE_TIME": "[REDACTED_DOB]",
    "PERSON": "[REDACTED_NAME]",
    "LOCATION": "[REDACTED_ADDRESS]",
    "PHONE_NUMBER": "[REDACTED_PHONE]",
    "EMAIL_ADDRESS": "[REDACTED_EMAIL]",
}

_ENTITIES_TO_DETECT = list(_ENTITY_OPERATOR_MAP.keys())

_SPACY_MODEL = "en_core_web_lg"


def _ensure_spacy_model() -> None:
    """Download the spaCy model if it is not already installed."""
    if spacy.util.is_package(_SPACY_MODEL):
        return
    # Build the wheel URL from spaCy's own download base URL
    from spacy.cli.download import get_compatibility

    compat = get_compatibility()
    version = compat[_SPACY_MODEL][0]
    wheel = f"{_SPACY_MODEL}-{version}-py3-none-any.whl"
    tag = f"{_SPACY_MODEL}-{version}"
    base_url = spacy.about.__download_url__
    if not base_url.endswith("/"):
        base_url += "/"
    url = f"{base_url}{tag}/{wheel}"
    # Prefer uv (used in uv-managed venvs), fall back to pip
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", url]
    else:
        cmd = [sys.executable, "-m", "pip", "install", url]
    subprocess.check_call(cmd)


class Redactor:
    """Detects and masks PII in text using Presidio.

    Replaces SSNs, dates of birth, person names, addresses, phone numbers,
    and email addresses with labelled placeholders such as
    ``[REDACTED_SSN]``.
    """

    def __init__(self) -> None:
        _ensure_spacy_model()
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operators = {
            entity: OperatorConfig("replace", {"new_value": placeholder})
            for entity, placeholder in _ENTITY_OPERATOR_MAP.items()
        }

    def redact(self, text: str) -> str:
        """Detect and replace PII in *text*.

        Args:
            text: The input text to redact.

        Returns:
            Text with PII replaced by labelled placeholders.
        """
        if not text:
            return text

        results = self._analyzer.analyze(
            text=text,
            entities=_ENTITIES_TO_DETECT,
            language="en",
        )

        if not results:
            return text

        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self._operators,
        )
        return anonymized.text
