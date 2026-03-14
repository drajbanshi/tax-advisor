"""Download and manage IRS reference documents."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from rich.console import Console

# Remote locations for bundled IRS reference markdowns.
# Keys are local filenames, values are public download URLs.
REFERENCE_DOCS: dict[str, str] = {
    "1040.md": "https://raw.githubusercontent.com/drajbanshi/tax-advisor/main/markdowns/1040.md",
    "p17.md": "https://raw.githubusercontent.com/drajbanshi/tax-advisor/main/markdowns/p17.md",
}


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
