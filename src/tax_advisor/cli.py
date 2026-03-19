"""Interactive REPL for the tax-advisor CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import is_done
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tax_advisor.agent import Agent
from tax_advisor.config import (
    PROVIDER_DEFAULT_MODELS,
    Settings,
    infer_provider_from_model,
    normalize_model_for_provider,
)
from tax_advisor.session import Session
from tax_advisor.tools import set_tool_settings


WELCOME_MESSAGE = """\
[bold green]tax-advisor[/bold green] — AI Tax Advisor Chat

Provider: [cyan]{provider}[/cyan]
Model: [cyan]{model}[/cyan]
Embedding: [cyan]{embedding_model}[/cyan]
Session: [cyan]{session_id}[/cyan]
Type your question and press [bold]Enter[/bold] to send. Paste multi-line text directly.

Commands:
  [bold]/quit[/bold]              Exit the chat
  [bold]/clear[/bold]             Clear conversation history
  [bold]/new[/bold]               Start a new session
  [bold]/sessions[/bold]          List saved sessions
  [bold]/continue <id>[/bold]     Resume a previous session
  [bold]/end-session[/bold]       Delete current session and start fresh
  [bold]/provider <name>[/bold]   [dim](anthropic|openai|bedrock|llama)[/dim]
  [bold]/model <name>[/bold]      Switch to a different model
  [bold]/ingest [path][/bold]     Ingest PDF/markdown docs into session [dim](default: documents/)[/dim]
  [bold]/ingest [path] --reference [--no-redact][/bold]  Ingest into IRS reference collection [dim](default: documents/)[/dim]
  [bold]/index[/bold]             Show vector index statistics
  [bold]/apikey[/bold]            Set or update your API key
  [bold]/reset[/bold]             Delete all data and start fresh
  [bold]/version[/bold]           Show version
"""


def _persist_to_env_file(env_file: Path, name: str, value: str) -> None:
    """Upsert *name=value* in the given .env file."""
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    replaced = False
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                lines.append(f"{name}={value}")
                replaced = True
            else:
                lines.append(line)
    if not replaced:
        lines.append(f"{name}={value}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _initial_setup(data_dir: Path, console: Console) -> None:
    """Prompt for provider and credentials on first run when no provider is configured."""
    env_file = data_dir / ".env"

    # Check if initial setup has already completed
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip() == "TAX_ADVISOR_INIT=true":
                return  # Already initialized

    console.print(
        "\n[bold green]Welcome to tax-advisor![/bold green]\n"
        "Let's set up your LLM provider.\n"
    )
    console.print("  [bold]1[/bold] — anthropic [dim](Claude, default)[/dim]")
    console.print("  [bold]2[/bold] — bedrock   [dim](AWS)[/dim]")
    console.print("  [bold]3[/bold] — llama     [dim](local Ollama)[/dim]")
    console.print()

    choice = console.input("Select provider [1/2/3]: ").strip()
    provider_map = {"1": "anthropic", "2": "bedrock", "3": "llama"}
    provider = provider_map.get(choice)

    if not provider:
        # Allow typing the provider name directly
        if choice.lower() in ("anthropic", "bedrock", "llama"):
            provider = choice.lower()
        else:
            console.print("[dim]Defaulting to anthropic.[/dim]\n")
            provider = "anthropic"

    _persist_to_env_file(env_file, "TAX_ADVISOR_PROVIDER", provider)
    os.environ["TAX_ADVISOR_PROVIDER"] = provider

    if provider == "anthropic":
        console.print(
            "\nAn API key is required for the Anthropic provider.\n"
            "You can get one at [link=https://console.anthropic.com/settings/keys]"
            "https://console.anthropic.com/settings/keys[/link]\n"
        )
        key = console.input("Enter your Anthropic API key: ").strip()
        if key:
            _persist_to_env_file(env_file, "ANTHROPIC_API_KEY", key)
            os.environ["ANTHROPIC_API_KEY"] = key
            console.print(
                f"\n[green]Provider set to anthropic. "
                f"API key saved to {env_file}[/green]\n"
            )
        else:
            console.print(
                "\n[yellow]No key provided. Use [bold]/apikey[/bold] to set it later.[/yellow]\n"
            )
    elif provider == "bedrock":
        profile = console.input("Enter your AWS profile name: ").strip()
        if profile:
            _persist_to_env_file(env_file, "TAX_ADVISOR_AWS_PROFILE", profile)
            os.environ["TAX_ADVISOR_AWS_PROFILE"] = profile
            console.print(
                f"\n[green]Provider set to bedrock "
                f"(profile: {profile}). Saved to {env_file}[/green]\n"
            )
        else:
            console.print(
                "\n[yellow]No profile provided. Set TAX_ADVISOR_AWS_PROFILE "
                "or use [bold]/provider bedrock <profile>[/bold] later.[/yellow]\n"
            )
    else:
        console.print(f"\n[green]Provider set to {provider}. Saved to {env_file}[/green]\n")


def _ensure_openai_api_key(settings: Settings, console: Console) -> None:
    """Prompt for an OpenAI API key if the provider needs one and none is set."""
    if settings.provider != "openai":
        return
    if settings.openai_api_key:
        return

    console.print(
        "\n[bold yellow]No OpenAI API key found.[/bold yellow]\n"
        "An API key is required for the OpenAI provider.\n"
        "You can get one at [link=https://platform.openai.com/api-keys]"
        "https://platform.openai.com/api-keys[/link]\n"
    )
    key = console.input("Enter your OpenAI API key: ").strip()
    if not key:
        console.print("[red]No key provided. Some features will not work.[/red]\n")
        return
    settings.set_openai_api_key(key)
    console.print(
        f"[green]API key saved to {settings.env_file}[/green]\n"
    )


def _ensure_anthropic_api_key(settings: Settings, console: Console) -> None:
    """Prompt for an Anthropic API key if the provider needs one and none is set."""
    if settings.provider != "anthropic":
        return
    if settings.anthropic_api_key:
        return

    console.print(
        "\n[bold yellow]No Anthropic API key found.[/bold yellow]\n"
        "An API key is required for the Anthropic provider.\n"
        "You can get one at [link=https://console.anthropic.com/settings/keys]"
        "https://console.anthropic.com/settings/keys[/link]\n"
    )
    key = console.input("Enter your Anthropic API key: ").strip()
    if not key:
        console.print("[red]No key provided. Some features will not work.[/red]\n")
        return
    settings.set_anthropic_api_key(key)
    console.print(
        f"[green]API key saved to {settings.env_file}[/green]\n"
    )


def _first_run_check(settings: Settings, console: Console) -> None:
    """Download IRS reference docs and offer to ingest them on first run."""
    env_file = settings.env_file
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip() == "TAX_ADVISOR_INIT=true":
                return

    # Ensure data directory exists
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    from tax_advisor.data import REFERENCE_DOCS, download_reference_docs

    if not REFERENCE_DOCS:
        _persist_to_env_file(settings.env_file, "TAX_ADVISOR_INIT", "true")
        return

    console.print(
        f"\n[bold]IRS reference documents available[/bold] "
        f"({len(REFERENCE_DOCS)} file(s))."
    )
    answer = console.input(
        "Download and ingest them into the vector store now? [Y/n] "
    ).strip().lower()

    if answer in ("", "y", "yes"):
        console.print()
        ref_dir = download_reference_docs(settings.data_dir, console)

        md_files = list(ref_dir.glob("*.md"))
        if md_files:
            from tax_advisor.rag.pipeline import ingest_documents

            console.print()
            ingested = ingest_documents(
                ref_dir,
                settings,
                console,
                skip_redaction=True,
            )
            if ingested > 0:
                _persist_to_env_file(settings.env_file, "TAX_ADVISOR_INIT", "true")
            else:
                console.print(
                    "[yellow]Ingestion failed. You will be prompted again "
                    "next time, or run [bold]/ingest --reference[/bold] "
                    "to retry.[/yellow]\n"
                )
        else:
            console.print(
                "[yellow]No documents were downloaded. You will be prompted "
                "again next time, or run [bold]/ingest --reference[/bold] "
                "to retry.[/yellow]\n"
            )
    else:
        console.print(
            "[dim]Skipped. Run [bold]/ingest --reference[/bold] "
            "any time to download and ingest.[/dim]\n"
        )
        _persist_to_env_file(settings.env_file, "TAX_ADVISOR_INIT", "true")


def main() -> None:
    """Entry point for the tax-advisor CLI."""
    from tax_advisor import __version__

    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"tax-advisor {__version__}")
        return

    from dotenv import load_dotenv

    # Load project-local .env first, then user-level .env (does not override)
    load_dotenv()

    console = Console()

    # Pre-create data dir so we can load its .env
    data_dir = Path(os.environ.get("TAX_ADVISOR_DATA_DIR", str(Path.home() / ".tax-advisor")))
    data_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(data_dir / ".env")

    # First-time provider setup (before Settings is created)
    _initial_setup(data_dir, console)

    settings = Settings()
    set_tool_settings(settings)

    # Prompt for API key if needed
    _ensure_anthropic_api_key(settings, console)
    _ensure_openai_api_key(settings, console)

    # Download IRS reference docs on first run (sets TAX_ADVISOR_INIT=true)
    _first_run_check(settings, console)

    # Create initial session
    active_session = Session.create(sessions_dir=settings.sessions_dir)
    settings.session_collection = active_session.collection_name

    agent = Agent(settings, console)

    console.print(
        Panel(
            WELCOME_MESSAGE.format(
                provider=settings.provider,
                model=settings.model,
                embedding_model=settings.embedding_model,
                session_id=active_session.id,
            ),
            title="Welcome",
            border_style="green",
        )
    )

    # Key bindings: Enter submits, Alt+Enter inserts newline.
    # Pasted text (bracketed paste) is inserted literally, so multi-line
    # pastes work without triggering submit.
    kb = KeyBindings()

    @kb.add("enter", filter=~is_done)
    def _submit(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        multiline=True,
        key_bindings=kb,
    )

    while True:
        try:
            user_input = session.prompt(
                "you> ",
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        text = user_input.strip()
        if not text:
            continue

        # Handle commands
        if text.startswith("/"):
            result = _handle_command(text, settings, agent, console, active_session)
            if result is None:
                # Quit signal
                break
            if isinstance(result, Session):
                # Session changed
                active_session = result
                settings.session_collection = active_session.collection_name
                continue
            # result is True — command handled, continue
            continue

        # Run the agent
        try:
            console.print()
            agent.run(text)
            console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]\n")
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]\n")


def _handle_command(
    text: str,
    settings: Settings,
    agent: Agent,
    console: Console,
    active_session: Session,
) -> bool | Session | None:
    """Handle slash commands.

    Returns:
        ``True`` if the command was handled normally.
        A :class:`Session` if the active session changed.
        ``None`` to signal the REPL should quit.
    """
    lower = text.lower()

    if lower == "/quit":
        console.print("[dim]Goodbye![/dim]")
        return None

    if lower == "/clear":
        agent.clear_history()
        console.print("[green]Conversation cleared.[/green]\n")
        return True

    if lower == "/version":
        from tax_advisor import __version__

        console.print(f"[cyan]tax-advisor {__version__}[/cyan]\n")
        return True

    # -- Session commands -----------------------------------------------------

    if lower == "/new":
        new_session = Session.create(sessions_dir=settings.sessions_dir)
        agent.clear_history()
        console.print(
            f"[green]New session started:[/green] [cyan]{new_session.id}[/cyan]\n"
        )
        return new_session

    if lower == "/sessions":
        sessions = Session.list_all(sessions_dir=settings.sessions_dir)
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]\n")
            return True
        for s in sessions:
            marker = " [bold green]← active[/bold green]" if s.id == active_session.id else ""
            console.print(
                f"  [cyan]{s.id}[/cyan]  "
                f"[dim]{s.created_at}[/dim]{marker}"
            )
        console.print()
        return True

    if lower.startswith("/continue"):
        parts = text.split()
        if len(parts) < 2:
            console.print("[yellow]Usage: /continue <session-id>[/yellow]\n")
            return True
        session_id = parts[1].strip()
        try:
            loaded = Session.load(session_id, sessions_dir=settings.sessions_dir)
        except FileNotFoundError:
            console.print(f"[red]Session not found: {session_id}[/red]\n")
            return True
        agent.clear_history()
        console.print(
            f"[green]Resumed session:[/green] [cyan]{loaded.id}[/cyan]\n"
        )
        return loaded

    if lower == "/end-session":
        # Delete the current session's ChromaDB collection and JSON file
        try:
            from tax_advisor.rag.pipeline import _build_session_index

            session_index = _build_session_index(settings)
            if session_index is not None:
                session_index.delete_collection()
        except Exception as exc:
            console.print(f"[yellow]Warning: could not delete collection: {exc}[/yellow]")
        active_session.delete_file()
        # Auto-create a replacement session
        new_session = Session.create(sessions_dir=settings.sessions_dir)
        agent.clear_history()
        console.print(
            f"[green]Session ended. New session:[/green] [cyan]{new_session.id}[/cyan]\n"
        )
        return new_session

    # -- Model/provider commands ----------------------------------------------

    if lower.startswith("/model"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            console.print(f"[dim]Current model:[/dim] [cyan]{settings.model}[/cyan]\n")
        else:
            new_model = normalize_model_for_provider(parts[1], settings.provider)
            settings.model = new_model
            inferred = infer_provider_from_model(new_model)
            if inferred:
                settings.provider = inferred
            console.print(f"[green]Switched to model:[/green] [cyan]{new_model}[/cyan]\n")
        return True

    if lower.startswith("/provider"):
        parts = text.split()
        if len(parts) < 2:
            active_profile = settings.bedrock_profile or "(not set)"
            console.print(
                "[dim]Current provider:[/dim] "
                f"[cyan]{settings.provider}[/cyan] "
                "[dim]| model:[/dim] "
                f"[cyan]{settings.model}[/cyan] "
                "[dim]| bedrock profile:[/dim] "
                f"[cyan]{active_profile}[/cyan]\n"
            )
            return True

        provider = parts[1].lower().strip()
        if provider not in PROVIDER_DEFAULT_MODELS:
            console.print(
                "[yellow]Unknown provider. Use one of: "
                "anthropic, openai, bedrock, llama[/yellow]\n"
            )
            return True

        if provider == "bedrock":
            profile = parts[2].strip() if len(parts) >= 3 else settings.bedrock_profile
            if not profile:
                console.print(
                    "[yellow]Bedrock requires an AWS profile. "
                    "Use `/provider bedrock <profile>` or set "
                    "`TAX_ADVISOR_AWS_PROFILE`.[/yellow]\n"
                )
                return True
            settings.bedrock_profile = profile
            settings.provider = provider
            settings.model = PROVIDER_DEFAULT_MODELS[provider]
            console.print(
                "[green]Switched provider:[/green] "
                f"[cyan]{provider}[/cyan] "
                "[dim](profile:[/dim] "
                f"[cyan]{profile}[/cyan][dim])[/dim] "
                "[dim]| model:[/dim] "
                f"[cyan]{settings.model}[/cyan]\n"
            )
            return True

        settings.provider = provider
        settings.model = PROVIDER_DEFAULT_MODELS[provider]
        console.print(
            "[green]Switched provider:[/green] "
            f"[cyan]{provider}[/cyan] "
            "[dim]| model:[/dim] "
            f"[cyan]{settings.model}[/cyan]\n"
        )
        return True

    # -- Template command -------------------------------------------------------

    if lower.startswith("/template"):
        parts = text.split()
        if len(parts) < 2 or parts[1].lower() not in ("w2", "1099"):
            console.print("[yellow]Usage: /template <w2|1099>[/yellow]\n")
            return True

        from tax_advisor.templates import generate_1099_template, generate_w2_template

        kind = parts[1].lower()
        if kind == "w2":
            path = generate_w2_template()
        else:
            path = generate_1099_template()
        console.print(f"[green]Template written:[/green] [cyan]{path}[/cyan]")
        console.print(f"Fill in the values, then run: [bold]/upload {path.name}[/bold]\n")
        return True

    # -- Upload (W-2 / 1099 vision extraction) command -------------------------

    if lower.startswith("/upload"):
        parts = text.split()
        if len(parts) < 2:
            console.print("[yellow]Usage: /upload <path> [path2 ...][/yellow]\n")
            return True

        # Collect all image and YAML paths from arguments (files and directories)
        image_extensions = {".png", ".jpg", ".jpeg"}
        yaml_extensions = {".yaml", ".yml"}
        all_images: list[Path] = []
        all_yaml: list[Path] = []
        for arg in parts[1:]:
            p = Path(arg.strip()).expanduser()
            if not p.exists():
                console.print(f"[red]Not found: {p}[/red]")
                continue
            if p.is_dir():
                found_images = sorted(
                    f for f in p.iterdir()
                    if f.suffix.lower() in image_extensions
                )
                found_yaml = sorted(
                    f for f in p.iterdir()
                    if f.suffix.lower() in yaml_extensions
                )
                if not found_images and not found_yaml:
                    console.print(f"[yellow]No images or YAML files found in {p}[/yellow]")
                else:
                    all_images.extend(found_images)
                    all_yaml.extend(found_yaml)
            elif p.suffix.lower() in image_extensions:
                all_images.append(p)
            elif p.suffix.lower() in yaml_extensions:
                all_yaml.append(p)
            else:
                console.print(f"[yellow]Skipping unsupported file: {p.name}[/yellow]")

        if not all_images and not all_yaml:
            console.print("[yellow]No valid files to process.[/yellow]\n")
            return True

        if all_yaml:
            console.print(f"Found [bold]{len(all_yaml)}[/bold] YAML file(s) to process.")
        if all_images:
            console.print(f"Found [bold]{len(all_images)}[/bold] image(s) to process.")
        console.print()

        try:
            # Process YAML files first
            if all_yaml:
                _process_yaml_uploads(all_yaml, settings, console)

            if not all_images:
                # Only YAML files — we're done
                console.print()
                return True

            from tax_advisor.form_classifier import classify_form

            # Classify each image
            groups: dict[str, list[Path]] = {"w2": [], "1099": [], "unknown": []}
            for img in all_images:
                console.print(f"Classifying [cyan]{img.name}[/cyan] …", end=" ")
                label = classify_form(
                    img,
                    model=settings.model,
                    bedrock_profile=settings.bedrock_profile,
                )
                groups[label].append(img)
                console.print(f"[bold]{label}[/bold]")

            console.print()

            # Report unknown images
            if groups["unknown"]:
                names = ", ".join(p.name for p in groups["unknown"])
                console.print(f"[yellow]Skipped (unknown form type): {names}[/yellow]\n")

            from tax_advisor.rag.pipeline import ingest_markdown_text

            # Process W-2 images (one per image)
            if groups["w2"]:
                from tax_advisor.w2_extract import extract_w2_from_image, w2_to_markdown

                for img in groups["w2"]:
                    console.print(f"Extracting W-2 from [cyan]{img.name}[/cyan] …")
                    w2 = extract_w2_from_image(
                        img,
                        model=settings.model,
                        bedrock_profile=settings.bedrock_profile,
                    )
                    _display_w2_table(w2, console)

                    md = w2_to_markdown(w2)
                    console.print("Ingesting W-2 into session collection …")
                    ingest_markdown_text(
                        md,
                        source_name=f"w2-{img.stem}",
                        settings=settings,
                        console=console,
                        collection_override=settings.session_collection,
                    )
                    console.print("[green]W-2 data extracted and indexed.[/green]\n")

            # Process 1099 images (all together in one call)
            if groups["1099"]:
                from tax_advisor.form1099_extract import (
                    extract_1099_from_images,
                    form1099_to_markdown,
                )

                names = ", ".join(p.name for p in groups["1099"])
                console.print(
                    f"Extracting consolidated 1099 from [cyan]{names}[/cyan] …"
                )
                data_1099 = extract_1099_from_images(
                    groups["1099"],
                    model=settings.model,
                    bedrock_profile=settings.bedrock_profile,
                )
                _display_1099_tables(data_1099, console)

                md = form1099_to_markdown(data_1099)
                console.print("Ingesting 1099 into session collection …")
                ingest_markdown_text(
                    md,
                    source_name="1099-consolidated",
                    settings=settings,
                    console=console,
                    collection_override=settings.session_collection,
                )
                console.print("[green]1099 data extracted and indexed.[/green]\n")

        except Exception as exc:
            console.print(f"[red]Upload error: {exc}[/red]")
        console.print()
        return True

    # -- Ingest command -------------------------------------------------------

    if lower.startswith("/ingest"):
        parts = text.split()
        is_reference = "--reference" in parts
        skip_redaction = "--no-redact" in parts
        # Remove flags so they aren't treated as paths
        path_parts = [
            p for p in parts[1:]
            if p not in ("--reference", "--no-redact")
        ]

        try:
            from tax_advisor.rag.pipeline import ingest_documents

            if is_reference:
                if path_parts:
                    docs_dir = Path(path_parts[0])
                else:
                    # No path given — download reference docs automatically
                    from tax_advisor.data import download_reference_docs

                    console.print("Downloading IRS reference documents …")
                    docs_dir = download_reference_docs(
                        settings.data_dir, console,
                    )
                    skip_redaction = True

                ingest_documents(
                    docs_dir, settings, console,
                    skip_redaction=skip_redaction,
                )
            else:
                docs_dir = Path(path_parts[0]) if path_parts else settings.docs_dir
                # Ingest into the session collection (PII redaction always on)
                ingest_documents(
                    docs_dir, settings, console,
                    skip_redaction=False,
                    collection_override=settings.session_collection,
                )
        except Exception as exc:
            console.print(f"[red]Ingestion error: {exc}[/red]")
        console.print()
        return True

    # -- Index command --------------------------------------------------------

    if lower == "/index":
        try:
            from tax_advisor.rag.pipeline import get_index_stats

            stats = get_index_stats(settings)
            ref = stats["reference"]
            console.print(
                f"[dim]Reference collection:[/dim] [cyan]{ref['collection']}[/cyan]  "
                f"[dim]Documents:[/dim] [cyan]{ref['document_count']}[/cyan]"
            )
            if "session" in stats:
                sess = stats["session"]
                console.print(
                    f"[dim]Session collection:[/dim]   [cyan]{sess['collection']}[/cyan]  "
                    f"[dim]Documents:[/dim] [cyan]{sess['document_count']}[/cyan]"
                )
            console.print()
        except Exception as exc:
            console.print(f"[red]Index error: {exc}[/red]\n")
        return True

    # -- API key command ------------------------------------------------------

    if lower == "/apikey":
        if settings.provider == "anthropic":
            if settings.anthropic_api_key:
                masked = settings.anthropic_api_key[:3] + "…" + settings.anthropic_api_key[-4:]
                console.print(f"[dim]Current Anthropic key:[/dim] [cyan]{masked}[/cyan]")
            else:
                console.print("[dim]No Anthropic API key is currently set.[/dim]")
            key = console.input("Enter new Anthropic API key (blank to keep current): ").strip()
            if key:
                settings.set_anthropic_api_key(key)
                console.print(f"[green]API key updated and saved to {settings.env_file}[/green]\n")
            else:
                console.print("[dim]Key unchanged.[/dim]\n")
        else:
            if settings.openai_api_key:
                masked = settings.openai_api_key[:3] + "…" + settings.openai_api_key[-4:]
                console.print(f"[dim]Current OpenAI key:[/dim] [cyan]{masked}[/cyan]")
            else:
                console.print("[dim]No OpenAI API key is currently set.[/dim]")
            key = console.input("Enter new OpenAI API key (blank to keep current): ").strip()
            if key:
                settings.set_openai_api_key(key)
                console.print(f"[green]API key updated and saved to {settings.env_file}[/green]\n")
            else:
                console.print("[dim]Key unchanged.[/dim]\n")
        return True

    # -- Reset command --------------------------------------------------------

    if lower == "/reset":
        import shutil

        console.print(
            f"[bold red]This will delete everything[/bold red] in "
            f"[cyan]{settings.data_dir}[/cyan]:\n"
            f"  • ChromaDB index (reference + session embeddings)\n"
            f"  • Downloaded reference documents\n"
            f"  • All saved sessions\n"
            f"  • First-run state (you will be prompted to set up again)\n\n"
            f"  [dim]Your API key in {settings.env_file} will be kept.[/dim]"
        )
        confirm = console.input("\nType [bold]yes[/bold] to confirm: ").strip().lower()
        if confirm != "yes":
            console.print("[dim]Reset cancelled.[/dim]\n")
            return True

        # Preserve the .env file
        env_backup = None
        if settings.env_file.exists():
            env_backup = settings.env_file.read_text(encoding="utf-8")

        # Wipe the data directory
        if settings.data_dir.exists():
            shutil.rmtree(settings.data_dir)
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        # Restore .env, but remove TAX_ADVISOR_INIT so setup runs again
        if env_backup is not None:
            restored_lines = [
                line for line in env_backup.splitlines()
                if not line.startswith("TAX_ADVISOR_INIT=")
            ]
            settings.env_file.write_text(
                "\n".join(restored_lines) + "\n", encoding="utf-8"
            )

        # Clear in-memory state
        agent.clear_history()
        settings.session_collection = None

        console.print(
            "[green]All data deleted.[/green] "
            "Restart tax-advisor to set up again.\n"
        )
        return None  # exit the REPL

    console.print(f"[yellow]Unknown command: {text}[/yellow]\n")
    return True


def _display_w2_table(w2: Any, console: Console) -> None:
    """Render W-2 extracted data as a Rich table for user verification."""
    table = Table(title="W-2 Extraction Results", show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    # Identifiers
    table.add_row("Employee SSN (a)", w2.employee_ssn)
    table.add_row("Employer EIN (b)", w2.employer_ein)
    table.add_row("Employer Name (c)", w2.employer_name)
    table.add_row("Employer Address", w2.employer_address)
    table.add_row("Control Number (d)", w2.control_number)
    table.add_row("Employee Name (e/f)", w2.employee_name)
    table.add_row("Employee Address", w2.employee_address)

    # Compensation & taxes
    table.add_row("Box 1 — Wages, tips, other comp.", w2.box1_wages)
    table.add_row("Box 2 — Federal income tax withheld", w2.box2_fed_tax_withheld)
    table.add_row("Box 3 — Social Security wages", w2.box3_ss_wages)
    table.add_row("Box 4 — SS tax withheld", w2.box4_ss_tax_withheld)
    table.add_row("Box 5 — Medicare wages", w2.box5_medicare_wages)
    table.add_row("Box 6 — Medicare tax withheld", w2.box6_medicare_tax_withheld)
    table.add_row("Box 7 — SS tips", w2.box7_ss_tips)
    table.add_row("Box 8 — Allocated tips", w2.box8_allocated_tips)
    table.add_row("Box 10 — Dependent care", w2.box10_dependent_care)
    table.add_row("Box 11 — Nonqualified plans", w2.box11_nonqualified_plans)

    # Box 12
    if w2.box12:
        for entry in w2.box12:
            table.add_row(f"Box 12 — Code {entry.code}", entry.amount)
    else:
        table.add_row("Box 12", "(none)")

    # Box 13
    table.add_row("Box 13 — Statutory employee", "Yes" if w2.box13_statutory_employee else "No")
    table.add_row("Box 13 — Retirement plan", "Yes" if w2.box13_retirement_plan else "No")
    table.add_row("Box 13 — Third-party sick pay", "Yes" if w2.box13_third_party_sick_pay else "No")

    # Box 14
    table.add_row("Box 14 — Other", w2.box14_other or "(none)")

    # State / local
    table.add_row("Box 15 — State", w2.box15_state)
    table.add_row("Box 15 — Employer state ID", w2.box15_employer_state_id)
    table.add_row("Box 16 — State wages", w2.box16_state_wages)
    table.add_row("Box 17 — State income tax", w2.box17_state_tax)
    table.add_row("Box 18 — Local wages", w2.box18_local_wages)
    table.add_row("Box 19 — Local income tax", w2.box19_local_tax)
    table.add_row("Box 20 — Locality name", w2.box20_locality_name)

    console.print()
    console.print(table)
    console.print()


def _display_1099_tables(data: Any, console: Console) -> None:
    """Render consolidated 1099 extracted data as Rich tables."""
    # Payer / Recipient info
    pi = data.payer_info
    info_table = Table(title="1099 — Payer / Recipient Info", show_lines=True)
    info_table.add_column("Field", style="bold")
    info_table.add_column("Value")
    info_table.add_row("Payer Name", pi.payer_name)
    info_table.add_row("Payer TIN", pi.payer_tin)
    info_table.add_row("Payer Address", pi.payer_address)
    info_table.add_row("Recipient Name", pi.recipient_name)
    info_table.add_row("Recipient TIN", pi.recipient_tin)
    info_table.add_row("Recipient Address", pi.recipient_address)
    info_table.add_row("Account Number", pi.account_number)
    if data.tax_year:
        info_table.add_row("Tax Year", data.tax_year)
    console.print()
    console.print(info_table)

    # 1099-B Summary
    if data.form_1099b is not None:
        b = data.form_1099b
        b_table = Table(title="1099-B — Proceeds Summary", show_lines=True)
        b_table.add_column("Category", style="bold")
        b_table.add_column("Proceeds")
        b_table.add_column("Cost Basis")
        b_table.add_column("Wash Sale Loss")
        b_table.add_column("Gain/Loss")
        b_table.add_row(
            "Short-Term",
            b.short_term_proceeds,
            b.short_term_cost_basis,
            b.short_term_wash_sale_loss,
            b.short_term_gain_loss,
        )
        b_table.add_row(
            "Long-Term",
            b.long_term_proceeds,
            b.long_term_cost_basis,
            b.long_term_wash_sale_loss,
            b.long_term_gain_loss,
        )
        if (
            b.undetermined_proceeds
            or b.undetermined_cost_basis
            or b.undetermined_wash_sale_loss
            or b.undetermined_gain_loss
        ):
            b_table.add_row(
                "Undetermined",
                b.undetermined_proceeds,
                b.undetermined_cost_basis,
                b.undetermined_wash_sale_loss,
                b.undetermined_gain_loss,
            )
        console.print()
        console.print(b_table)

    # 1099-DIV
    if data.form_1099div is not None:
        d = data.form_1099div
        d_table = Table(title="1099-DIV — Dividends and Distributions", show_lines=True)
        d_table.add_column("Field", style="bold")
        d_table.add_column("Value")
        d_table.add_row("Box 1a — Total ordinary dividends", d.box_1a_total_ordinary_dividends)
        d_table.add_row("Box 1b — Qualified dividends", d.box_1b_qualified_dividends)
        d_table.add_row("Box 2a — Total capital gain dist.", d.box_2a_total_capital_gain_dist)
        d_table.add_row("Box 2b — Unrecap. Sec. 1250 gain", d.box_2b_unrecap_sec_1250_gain)
        d_table.add_row("Box 2c — Sec. 1202 gain", d.box_2c_sec_1202_gain)
        d_table.add_row("Box 2d — Collectibles (28%) gain", d.box_2d_collectibles_gain)
        d_table.add_row("Box 2e — Sec. 897 ordinary dividends", d.box_2e_sec_897_ordinary_dividends)
        d_table.add_row("Box 2f — Sec. 897 capital gain", d.box_2f_sec_897_capital_gain)
        d_table.add_row("Box 3 — Nondividend distributions", d.box_3_nondividend_distributions)
        d_table.add_row("Box 4 — Federal tax withheld", d.box_4_federal_tax_withheld)
        d_table.add_row("Box 5 — Sec. 199A dividends", d.box_5_sec_199a_dividends)
        d_table.add_row("Box 6 — Investment expenses", d.box_6_investment_expenses)
        d_table.add_row("Box 7 — Foreign tax paid", d.box_7_foreign_tax_paid)
        d_table.add_row("Box 8 — Foreign country", d.box_8_foreign_country)
        d_table.add_row("Box 11 — Exempt-interest dividends", d.box_11_exempt_interest_dividends)
        d_table.add_row("Box 12 — Specified PAB interest div.", d.box_12_specified_pab_interest_dividends)
        d_table.add_row("Box 13 — State", d.box_13_state)
        d_table.add_row("Box 14 — State tax withheld", d.box_14_state_tax_withheld)
        d_table.add_row("Box 15 — State ID number", d.box_15_state_id_number)
        console.print()
        console.print(d_table)

    # 1099-INT
    if data.form_1099int is not None:
        i = data.form_1099int
        i_table = Table(title="1099-INT — Interest Income", show_lines=True)
        i_table.add_column("Field", style="bold")
        i_table.add_column("Value")
        i_table.add_row("Box 1 — Interest income", i.box_1_interest_income)
        i_table.add_row("Box 2 — Early withdrawal penalty", i.box_2_early_withdrawal_penalty)
        i_table.add_row("Box 3 — US savings bond interest", i.box_3_us_savings_bond_interest)
        i_table.add_row("Box 4 — Federal tax withheld", i.box_4_federal_tax_withheld)
        i_table.add_row("Box 8 — Tax-exempt interest", i.box_8_tax_exempt_interest)
        i_table.add_row("Box 10 — Market discount", i.box_10_market_discount)
        i_table.add_row("Box 11 — Bond premium", i.box_11_bond_premium)
        i_table.add_row("Box 12 — Bond premium (Treasury)", i.box_12_bond_premium_treasury)
        i_table.add_row("Box 13 — Bond premium (tax-exempt)", i.box_13_bond_premium_tax_exempt)
        i_table.add_row("Box 15 — State", i.box_15_state)
        i_table.add_row("Box 16 — State tax withheld", i.box_16_state_tax_withheld)
        i_table.add_row("Box 17 — State ID number", i.box_17_state_id_number)
        console.print()
        console.print(i_table)

    console.print()


def _process_yaml_uploads(
    yaml_files: list[Path],
    settings: Settings,
    console: Console,
) -> None:
    """Process YAML template files through the parse -> display -> markdown -> ingest pipeline."""
    from tax_advisor.form1099_extract import _parse_1099_data, form1099_to_markdown
    from tax_advisor.rag.pipeline import ingest_markdown_text
    from tax_advisor.templates import load_yaml_template
    from tax_advisor.w2_extract import _parse_w2_data, w2_to_markdown

    for yf in yaml_files:
        console.print(f"Processing YAML template [cyan]{yf.name}[/cyan] …")
        try:
            form_type, data = load_yaml_template(yf)
        except (ValueError, FileNotFoundError) as exc:
            console.print(f"[red]Error loading {yf.name}: {exc}[/red]")
            continue

        if form_type == "w2":
            w2 = _parse_w2_data(data)
            _display_w2_table(w2, console)
            md = w2_to_markdown(w2)
            console.print("Ingesting W-2 into session collection …")
            ingest_markdown_text(
                md,
                source_name=f"w2-{yf.stem}",
                settings=settings,
                console=console,
                collection_override=settings.session_collection,
            )
            console.print("[green]W-2 data extracted and indexed.[/green]\n")

        elif form_type == "1099":
            data_1099 = _parse_1099_data(data)
            _display_1099_tables(data_1099, console)
            md = form1099_to_markdown(data_1099)
            console.print("Ingesting 1099 into session collection …")
            ingest_markdown_text(
                md,
                source_name=f"1099-{yf.stem}",
                settings=settings,
                console=console,
                collection_override=settings.session_collection,
            )
            console.print("[green]1099 data extracted and indexed.[/green]\n")

        else:
            console.print(
                f"[yellow]Could not determine form type for {yf.name} — skipping.[/yellow]\n"
            )
