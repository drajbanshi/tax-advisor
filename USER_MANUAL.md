# Tax-Advisor User Manual

## Prerequisites

- macOS, Linux, or Windows (WSL)
- [mise](https://mise.jdx.dev/) (recommended) or Python 3.11+ installed manually

---

## Quick Install (Recommended)

If you have Python 3.11+ already installed (or want the script to check for you), run the one-liner:

```bash
curl -fsSL https://raw.githubusercontent.com/drajbanshi/tax-advisor/main/install.sh | bash
```

Or download and run manually:

```bash
curl -fsSL -o install.sh https://raw.githubusercontent.com/drajbanshi/tax-advisor/main/install.sh
chmod +x install.sh
./install.sh
```

The install script will:

1. Find a suitable Python (>= 3.11) on your system
2. Create a virtual environment at `~/.tax-advisor/venv`
3. Install `tax-advisor` from PyPI
4. Download the required spaCy language model
5. Create a `tax-advisor` wrapper script at `~/.tax-advisor/bin/`

After installation, add `~/.tax-advisor/bin` to your PATH (the script will print the exact command), then run:

```bash
tax-advisor
```

If the script reports that Python 3.11+ is not found, follow the manual steps below to install it first.

---

## 1. Install Python 3.11 with mise

[mise](https://mise.jdx.dev/) is a polyglot runtime manager that makes it easy to install and switch between Python versions.

### Install mise

```bash
curl https://mise.run | sh
```

Follow the instructions printed by the installer to add mise to your shell profile. For example, for bash:

```bash
echo 'eval "$(~/.local/bin/mise activate bash)"' >> ~/.bashrc
source ~/.bashrc
```

For zsh:

```bash
echo 'eval "$(~/.local/bin/mise activate zsh)"' >> ~/.zshrc
source ~/.zshrc
```

### Install Python 3.11

```bash
mise use python@3.11
```

This installs Python 3.11 and sets it as the active version in the current directory. Verify:

```bash
python --version
# Python 3.11.x
```

To set Python 3.11 globally (all directories):

```bash
mise use -g python@3.11
```

---

## 2. Create a Virtual Environment

Create and activate a virtual environment:

```bash
mkdir tax-advisor && cd tax-advisor
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (WSL / Git Bash)
source .venv/Scripts/activate
```

Verify the virtual environment is active:

```bash
which python
# Should point to .venv/bin/python
```

---

## 3. Install tax-advisor

With the virtual environment activated, install the package using pip:

```bash
pip install tax-advisor
```

To include optional AWS Bedrock support:

```bash
pip install "tax-advisor[bedrock]"
```

After installation, the `tax-advisor` command is available:

```bash
tax-advisor
```

You can also run it as a Python module:

```bash
python -m tax_advisor
```

---

## 4. Configure Your OpenAI API Key

tax-advisor uses the OpenAI API by default (provider: `openai`, model: `gpt-4o`). You need a valid OpenAI API key.

### Option A: Interactive prompt (recommended for first-time setup)

Simply run `tax-advisor`. If no API key is found, you will be prompted:

```
No OpenAI API key found.
An API key is required for the OpenAI provider.
You can get one at https://platform.openai.com/api-keys

Enter your OpenAI API key:
```

The key is saved to `~/.tax-advisor/.env` and loaded automatically on future runs.

### Option B: Environment variable

Export the key in your shell before running:

```bash
export OPENAI_API_KEY="sk-..."
tax-advisor
```

To make it permanent, add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

### Option C: Project `.env` file

Create or edit a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

tax-advisor loads this file automatically on startup.

### Option D: Update the key at any time

Inside the running CLI, use the `/apikey` command:

```
you> /apikey
Current key: sk-…1234
Enter new OpenAI API key (blank to keep current):
```

The updated key is saved to `~/.tax-advisor/.env`.

---

## 5. Additional Configuration

All settings can be controlled via environment variables. Set them in your shell, in a `.env` file in the project root, or in `~/.tax-advisor/.env`.

| Variable | Description | Default |
|---|---|---|
| `TAX_ADVISOR_PROVIDER` | LLM provider (`openai`, `bedrock`, `llama`) | `openai` |
| `TAX_ADVISOR_MODEL` | Model identifier | `gpt-4o` |
| `TAX_ADVISOR_TEMPERATURE` | Sampling temperature | `0.3` |
| `TAX_ADVISOR_AWS_PROFILE` | AWS profile for Bedrock provider | _(none)_ |
| `TAX_ADVISOR_DATA_DIR` | User data directory | `~/.tax-advisor` |
| `TAX_ADVISOR_DOCS_DIR` | Document directory for ingestion | `./documents` |
| `TAX_ADVISOR_CHROMA_DIR` | ChromaDB storage directory | `~/.tax-advisor/chroma_db` |
| `TAX_ADVISOR_CHROMA_COLLECTION` | Default vector collection name | `tax_documents` |
| `TAX_ADVISOR_EMBEDDING_MODEL` | Embedding model | Provider-specific default |
| `TAX_ADVISOR_SYSTEM_PROMPT` | Custom system prompt | _(built-in)_ |
| `OPENAI_API_KEY` | OpenAI API key | _(none)_ |

---

## 6. CLI Commands Reference

Once inside the tax-advisor REPL, the following slash commands are available:

| Command | Description |
|---|---|
| `/quit` | Exit the chat |
| `/clear` | Clear conversation history |
| `/new` | Start a new session |
| `/sessions` | List saved sessions |
| `/continue <id>` | Resume a previous session |
| `/end-session` | Delete current session and start fresh |
| `/provider <name>` | Switch provider (`openai`, `bedrock`, `llama`) |
| `/model <name>` | Switch to a different model |
| `/ingest [path]` | Ingest PDF/markdown docs into session (default: `documents/`) |
| `/ingest [path] --reference [--no-redact]` | Ingest into IRS reference collection |
| `/index` | Show vector index statistics |
| `/apikey` | Set or update your OpenAI API key |
| `/reset` | Delete all data and start fresh |

Type your question at the `you>` prompt and press **Enter**. Use **Alt+Enter** for multi-line input.

---

## 7. First Run

On the first launch, tax-advisor will offer to download and ingest IRS reference documents into the vector store. This provides the AI with a knowledge base of official tax guidance. Type `y` (or press Enter) to proceed, or `n` to skip. You can always run `/ingest --reference` later to trigger this manually.

---

## 8.Sample Questions
1. Whats the standard deduction for the year 2025?
2. Whats the SALT limit for year 2025?
3. Prepare my taxes with provided information: copy paste filled prompt [earning](prepare_my_tax.md))
