"""Settings and configuration for tax-advisor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "anthropic/claude-opus-4-6-20250219"
DEFAULT_TEMPERATURE = 0.3
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "anthropic/claude-opus-4-6-20250219",
    "openai": "openai/gpt-4o",
    "bedrock": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "llama": "ollama/llama3.1",
}

DEFAULT_DATA_DIR = Path.home() / ".tax-advisor"
DEFAULT_DOCS_DIR = "./documents"
DEFAULT_CHROMA_COLLECTION = "tax_documents"

SYSTEM_PROMPT = """\
You are an expert tax advisor AI assistant. You help users understand tax \
concepts, answer questions about tax filing, deductions, credits, and tax \
planning strategies. You can look up the current date when needed. You know \
all the tax rules for tax year 2025 from indexed Graph RAG.

You can help prepare taxes, 1040 forms, require schedules and worksheets.

When users ask about specific tax rules, forms, publications, or IRS guidance, \
use the search_tax_documents tool to find relevant information from indexed \
IRS documents. Ground your answers in the retrieved content and cite the \
source document (filename, section) when possible.

Always clarify that you provide general tax information and that users should \
consult a qualified tax professional for advice specific to their situation.

When providing information, cite relevant tax forms, schedules, or IRS \
publications where appropriate.
"""


def infer_provider_from_model(model: str) -> str | None:
    """Infer provider from a model identifier prefix."""
    lowered = model.lower()
    if lowered.startswith("anthropic/"):
        return "anthropic"
    if lowered.startswith("openai/"):
        return "openai"
    if lowered.startswith("bedrock/"):
        return "bedrock"
    if lowered.startswith("ollama/"):
        return "llama"
    return None


def normalize_model_for_provider(model: str, provider: str) -> str:
    """Normalize model IDs for provider-specific conventions."""
    normalized = model.strip()
    if normalized and "/" not in normalized:
        if provider == "anthropic":
            return f"anthropic/{normalized}"
        if provider == "bedrock":
            return f"bedrock/{normalized}"
        if provider == "openai":
            return f"openai/{normalized}"
    return normalized


def _default_embedding_model(provider: str) -> str:
    """Return provider-appropriate default embedding model."""
    from tax_advisor.rag.embeddings import default_embedding_model

    return default_embedding_model(provider)


_PROVIDER_EMBEDDING_PREFIXES: dict[str, list[str]] = {
    "bedrock": ["amazon."],
    "openai": ["text-embedding-"],
    "anthropic": ["local"],
    "llama": ["local"],
}


def _validate_embedding_model(model: str, provider: str) -> str:
    """Return model if it's compatible with provider, otherwise the provider default."""
    if model == "local":
        return model
    prefixes = _PROVIDER_EMBEDDING_PREFIXES.get(provider, [])
    if prefixes and not any(model.startswith(p) for p in prefixes):
        return _default_embedding_model(provider)
    return model


@dataclass
class Settings:
    """Application settings, loaded from environment variables with defaults."""

    provider: str = field(
        default_factory=lambda: os.environ.get("TAX_ADVISOR_PROVIDER", DEFAULT_PROVIDER)
    )
    model: str = field(
        default_factory=lambda: os.environ.get("TAX_ADVISOR_MODEL", "")
    )
    bedrock_profile: str | None = field(
        default_factory=lambda: os.environ.get("TAX_ADVISOR_AWS_PROFILE")
        or os.environ.get("AWS_PROFILE")
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get("TAX_ADVISOR_TEMPERATURE", str(DEFAULT_TEMPERATURE))
        )
    )
    system_prompt: str = field(
        default_factory=lambda: os.environ.get(
            "TAX_ADVISOR_SYSTEM_PROMPT", SYSTEM_PROMPT
        )
    )
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("TAX_ADVISOR_DATA_DIR", str(DEFAULT_DATA_DIR))
        )
    )
    docs_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("TAX_ADVISOR_DOCS_DIR", DEFAULT_DOCS_DIR)
        )
    )
    chroma_dir: Path = field(default=None)  # type: ignore[assignment]
    chroma_collection: str = field(
        default_factory=lambda: os.environ.get(
            "TAX_ADVISOR_CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("TAX_ADVISOR_EMBEDDING_MODEL", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    session_collection: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Normalize provider/model defaults after env loading."""
        self.provider = self.provider.lower().strip() or DEFAULT_PROVIDER
        if self.provider not in PROVIDER_DEFAULT_MODELS:
            self.provider = DEFAULT_PROVIDER

        if self.model:
            self.model = normalize_model_for_provider(self.model, self.provider)
            inferred = infer_provider_from_model(self.model)
            if inferred:
                self.provider = inferred
        else:
            self.model = PROVIDER_DEFAULT_MODELS.get(self.provider, DEFAULT_MODEL)

        # Resolve data_dir and derived paths
        self.data_dir = self.data_dir.expanduser()

        if self.chroma_dir is None:
            env_val = os.environ.get("TAX_ADVISOR_CHROMA_DIR")
            self.chroma_dir = Path(env_val) if env_val else self.data_dir / "chroma_db"

        # Provider-aware embedding model default
        if not self.embedding_model:
            self.embedding_model = _default_embedding_model(self.provider)
        else:
            self.embedding_model = _validate_embedding_model(self.embedding_model, self.provider)

    @property
    def sessions_dir(self) -> Path:
        """Directory for persisted session JSON files."""
        return self.data_dir / "sessions"

    @property
    def reference_docs_dir(self) -> Path:
        """Directory for downloaded IRS reference documents."""
        return self.data_dir / "reference"

    @property
    def initialized_sentinel(self) -> Path:
        """Sentinel file indicating first-run ingestion has been offered."""
        return self.data_dir / ".initialized"

    @property
    def env_file(self) -> Path:
        """Path to the persistent user ``.env`` file in the data directory."""
        return self.data_dir / ".env"

    def set_openai_api_key(self, key: str) -> None:
        """Store *key* in memory, the process environment, and the user env file."""
        self.openai_api_key = key
        os.environ["OPENAI_API_KEY"] = key
        self._persist_env_var("OPENAI_API_KEY", key)

    def set_anthropic_api_key(self, key: str) -> None:
        """Store *key* in memory, the process environment, and the user env file."""
        self.anthropic_api_key = key
        os.environ["ANTHROPIC_API_KEY"] = key
        self._persist_env_var("ANTHROPIC_API_KEY", key)

    def _persist_env_var(self, name: str, value: str) -> None:
        """Upsert *name=value* in :pyattr:`env_file`."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        env_path = self.env_file
        lines: list[str] = []
        replaced = False
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{name}="):
                    lines.append(f"{name}={value}")
                    replaced = True
                else:
                    lines.append(line)
        if not replaced:
            lines.append(f"{name}={value}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
