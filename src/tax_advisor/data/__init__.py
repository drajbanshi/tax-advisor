"""Bundled IRS reference documents shipped with the package."""

from __future__ import annotations

import urllib.request
from importlib import resources
from pathlib import Path

from rich.console import Console

# Remote locations for IRS reference markdowns (used by /ingest --reference).
# Keys are local filenames, values are public download URLs.
REFERENCE_DOCS: dict[str, str] = {
    "1040.md": "https://raw.githubusercontent.com/drajbanshi/tax-advisor/main/markdowns/1040.md",
    "p17.md": "https://raw.githubusercontent.com/drajbanshi/tax-advisor/main/markdowns/p17.md",
}


def bundled_reference_dir() -> Path:
    """Return the path to the reference docs bundled inside the package."""
    return Path(str(resources.files("tax_advisor.data") / "reference"))


def get_reference_docs_dir(data_dir: Path) -> Path:
    """Return the local directory where reference docs are stored."""
    return data_dir / "reference"


def download_reference_docs(
    data_dir: Path,
    console: Console,
) -> Path:
    """Download IRS reference markdowns into ``<data_dir>/reference/``.

    Skips files that already exist locally.  Returns the directory path.
    """
    dest = get_reference_docs_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for filename, url in REFERENCE_DOCS.items():
        target = dest / filename
        if target.exists():
            console.print(f"  [dim]{filename} already present — skipped[/dim]")
            continue
        console.print(f"  Downloading [cyan]{filename}[/cyan] …", end=" ")
        try:
            urllib.request.urlretrieve(url, target)
            size_kb = target.stat().st_size / 1024
            console.print(f"[green]OK[/green] ({size_kb:.0f} KB)")
        except Exception as exc:
            console.print(f"[red]failed: {exc}[/red]")

    return dest
