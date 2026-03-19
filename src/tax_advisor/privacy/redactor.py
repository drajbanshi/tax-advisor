"""PII redaction using Microsoft Presidio."""

from __future__ import annotations

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


class Redactor:
    """Detects and masks PII in text using Presidio.

    Replaces SSNs, dates of birth, person names, addresses, phone numbers,
    and email addresses with labelled placeholders such as
    ``[REDACTED_SSN]``.
    """

    def __init__(self) -> None:
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
