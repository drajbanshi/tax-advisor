"""Session management for user-uploaded document collections."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Session:
    """A user session backed by a per-session ChromaDB collection.

    Attributes:
        id: 12-character hex identifier.
        created_at: ISO-8601 timestamp of creation.
    """

    id: str
    created_at: str

    # Set by callers; defaults to legacy path for backwards compat.
    _sessions_dir: Path = Path(".sessions")

    @property
    def collection_name(self) -> str:
        """Derived ChromaDB collection name."""
        return f"session_{self.id}"

    # -- Persistence ----------------------------------------------------------

    def save(self) -> None:
        """Persist session metadata to ``<sessions_dir>/<id>.json``."""
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self._sessions_dir / f"{self.id}.json"
        path.write_text(
            json.dumps({"id": self.id, "created_at": self.created_at}),
            encoding="utf-8",
        )

    def delete_file(self) -> None:
        """Remove the persisted JSON file for this session."""
        path = self._sessions_dir / f"{self.id}.json"
        path.unlink(missing_ok=True)

    # -- Class helpers --------------------------------------------------------

    @classmethod
    def create(cls, sessions_dir: Path | None = None) -> Session:
        """Create a new session with a random 12-char hex ID."""
        session = cls(
            id=secrets.token_hex(6),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        if sessions_dir is not None:
            session._sessions_dir = sessions_dir
        session.save()
        return session

    @classmethod
    def load(cls, session_id: str, sessions_dir: Path | None = None) -> Session:
        """Load a session from its JSON file.

        Raises:
            FileNotFoundError: If the session file does not exist.
        """
        sdir = sessions_dir or Path(".sessions")
        path = sdir / f"{session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        session = cls(id=data["id"], created_at=data["created_at"])
        session._sessions_dir = sdir
        return session

    @classmethod
    def list_all(cls, sessions_dir: Path | None = None) -> list[Session]:
        """Return all saved sessions, sorted by creation time."""
        sdir = sessions_dir or Path(".sessions")
        if not sdir.exists():
            return []
        sessions: list[Session] = []
        for path in sorted(sdir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = cls(id=data["id"], created_at=data["created_at"])
                session._sessions_dir = sdir
                sessions.append(session)
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions
