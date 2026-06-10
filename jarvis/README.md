# Jarvis

Local AI agent accessible via email, powered by Ollama and Qwen.

## How It Works

Jarvis polls `jarvisledbetter@gmail.com` for incoming emails, runs them through a local Qwen model with agentic filesystem tools, and replies. Only emails from authorized senders are processed.

The agent can list directories, read files, search code, and show directory trees — all sandboxed to registered folders.

## Setup

```bash
# Install (editable, from repo root)
pip install -e .

# Configure credentials and model
jarvis configure

# Register folders for Jarvis to monitor
jarvis register /path/to/project
jarvis register .  # current directory
```

## CLI Commands

| Command | Description |
|---|---|
| `jarvis status` | Show which folders Jarvis is live in |
| `jarvis ask "question"` | Ask Jarvis directly from terminal |
| `jarvis register /path` | Register a folder (default: cwd) |
| `jarvis unregister /path` | Remove a folder |
| `jarvis list` | List all registered folders |
| `jarvis start` | Start the email polling service |
| `jarvis start -v` | Start with verbose logging |
| `jarvis configure` | Set Gmail credentials and model |
| `jarvis install-service` | Generate macOS launchd plist for auto-start |

## Running the Service

### Foreground (manual)

```bash
jarvis start
```

### Background (persistent, survives reboots)

```bash
jarvis install-service
launchctl load ~/Library/LaunchAgents/com.jarvis.agent.plist
```

To stop:

```bash
launchctl unload ~/Library/LaunchAgents/com.jarvis.agent.plist
```

## Email Usage

Send an email to `jarvisledbetter@gmail.com` from an authorized sender (default: `dledbetter456@gmail.com`). Jarvis will reply with the agent's response.

Example emails:
- "What folders are you currently live in?"
- "Show me the directory structure"
- "Search for 'attention' in the codebase"
- "Read the file v12/README.md"

## Configuration

Stored at `~/.jarvis/config.json`:

| Key | Default | Description |
|---|---|---|
| `model` | `qwen3.5:0.8b` | Ollama model to use |
| `poll_interval_seconds` | `60` | How often to check for new emails |
| `max_tool_iterations` | `5` | Max agent tool calls per email |
| `max_file_lines` | `200` | Max lines when reading a file |
| `authorized_senders` | `["dledbetter456@gmail.com"]` | Who can email Jarvis |
| `registered_folders` | `[]` | Folders Jarvis can access |

## Agent Tools

The agent has access to these tools (sandboxed to registered folders):

- **get_status** — List registered folders, file counts, and model info
- **list_directory** — List files and subdirectories at a path
- **read_file** — Read file contents (auto-parses `.ipynb` notebooks with outputs)
- **read_notebook** — Parse a Jupyter notebook: shows code cells, training logs, metrics, and errors
- **find_notebooks** — Find all `.ipynb` files across registered folders, with cell/output counts
- **search_files** — Grep for a string across code files
- **tree** — Show directory tree (default depth: 2)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a model pulled (e.g. `ollama pull qwen3.5:0.8b`)
- Gmail account with an [app password](https://myaccount.google.com/apppasswords) and IMAP enabled

## Logs

Service logs are written to `~/.jarvis/jarvis.log`.
