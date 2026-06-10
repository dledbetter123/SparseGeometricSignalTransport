"""Filesystem tools that the Jarvis agent can call."""

import json
import logging
import os
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from jarvis.config import is_path_allowed

logger = logging.getLogger("jarvis")


def get_status(config: dict) -> str:
    """List all registered folders and basic stats."""
    folders = config.get("registered_folders", [])
    if not folders:
        return "No folders registered. Use `jarvis register <path>` to add one."
    lines = [f"Jarvis is live in {len(folders)} folder(s):\n"]
    for folder in folders:
        if os.path.isdir(folder):
            count = sum(1 for _ in Path(folder).rglob("*") if _.is_file())
            lines.append(f"  - {folder}  ({count} files)")
        else:
            lines.append(f"  - {folder}  (NOT FOUND)")
    lines.append(f"\nModel: {config.get('model', 'unknown')}")
    return "\n".join(lines)


def list_directory(path: str, config: dict) -> str:
    """List files and directories at the given path."""
    if not is_path_allowed(path, config):
        return f"Access denied: '{path}' is not within a registered folder."
    if not os.path.isdir(path):
        return f"Not a directory: '{path}'"
    entries = sorted(os.listdir(path))
    lines = []
    for entry in entries[:100]:  # cap at 100 entries
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            lines.append(f"  [dir]  {entry}/")
        else:
            size = os.path.getsize(full)
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size // 1024}K"
            else:
                size_str = f"{size // (1024 * 1024)}M"
            lines.append(f"  [file] {entry}  ({size_str})")
    header = f"Contents of {path}  ({len(entries)} items):\n"
    if len(entries) > 100:
        header += "(showing first 100)\n"
    return header + "\n".join(lines)


def read_file(path: str, config: dict, max_lines: int = 0) -> str:
    """Read file contents, limited to max_lines. Automatically parses .ipynb notebooks."""
    if not is_path_allowed(path, config):
        return f"Access denied: '{path}' is not within a registered folder."
    if not os.path.isfile(path):
        return f"Not a file: '{path}'"
    # Route notebooks to the notebook reader
    if path.endswith(".ipynb"):
        return read_notebook(path, config)
    if max_lines <= 0:
        max_lines = config.get("max_file_lines", 200)
    try:
        with open(path, "r", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... (truncated at {max_lines} lines)")
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


def _save_notebook_via_jupyter(path: str):
    """Try to save a notebook by hitting the Jupyter server API before reading.
    This ensures we get the latest outputs if the notebook is still running."""
    abs_path = os.path.abspath(path)
    try:
        result = subprocess.run(
            ["jupyter", "server", "list", "--json"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            return  # No servers running
    except Exception:
        return

    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            server = json.loads(line)
        except json.JSONDecodeError:
            continue

        root_dir = server.get("root_dir", server.get("notebook_dir", ""))
        if not root_dir or not abs_path.startswith(root_dir):
            continue

        # Build the API path
        rel_path = os.path.relpath(abs_path, root_dir)
        base_url = server.get("url", "").rstrip("/")
        token = server.get("token", "")
        api_url = f"{base_url}/api/contents/{rel_path}"

        try:
            # POST a save request
            save_data = json.dumps({
                "type": "notebook",
                "content": json.load(open(abs_path)),
            }).encode()
            req = urllib.request.Request(
                api_url,
                data=save_data,
                method="PUT",
                headers={
                    "Authorization": f"token {token}",
                    "Content-Type": "application/json",
                },
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"Saved notebook via Jupyter API: {rel_path}")
        except Exception as e:
            # Also try saving via osascript (Cmd+S) as fallback
            logger.debug(f"Jupyter API save failed ({e}), trying osascript")
            try:
                subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to keystroke "s" using command down'],
                    timeout=3, capture_output=True
                )
            except Exception:
                pass
        return  # Only try the first matching server


def read_notebook(path: str, config: dict) -> str:
    """Parse a Jupyter notebook and extract code, outputs, and metrics.
    Attempts to save the notebook first to capture latest results."""
    if not is_path_allowed(path, config):
        return f"Access denied: '{path}' is not within a registered folder."
    if not os.path.isfile(path):
        return f"Not a file: '{path}'"
    # Try to save the notebook first to get latest outputs
    _save_notebook_via_jupyter(path)
    try:
        with open(path, "r", errors="replace") as f:
            nb = json.load(f)
    except Exception as e:
        return f"Error parsing notebook: {e}"

    cells = nb.get("cells", [])
    sections = []
    sections.append(f"Notebook: {os.path.basename(path)}")
    sections.append(f"Total cells: {len(cells)}")
    sections.append("")

    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])

        if cell_type == "markdown":
            # Only include headers and short markdown
            header_lines = [l for l in source.split("\n") if l.startswith("#")]
            if header_lines:
                sections.append(header_lines[0])
            continue

        # Code cell
        # Show a compact version of the source
        source_lines = source.strip().split("\n")
        source_preview = source_lines[0][:100] if source_lines else ""
        sections.append(f"--- Cell [{i}] code: {source_preview}")

        if not outputs:
            sections.append("  (no output)")
            continue

        for out in outputs:
            otype = out.get("output_type", "")
            if otype == "stream":
                text = "".join(out.get("text", []))
                # Show last 600 chars of stream output (where final metrics usually are)
                if len(text) > 600:
                    text = "...\n" + text[-600:]
                for line in text.strip().split("\n"):
                    sections.append(f"  > {line}")
            elif otype == "execute_result":
                text = "".join(out.get("data", {}).get("text/plain", []))
                if text:
                    for line in text.strip().split("\n")[:20]:
                        sections.append(f"  => {line}")
            elif otype == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                sections.append(f"  ERROR: {ename}: {evalue}")
            elif otype == "display_data":
                data_keys = list(out.get("data", {}).keys())
                if "text/plain" in data_keys:
                    text = "".join(out["data"]["text/plain"])
                    sections.append(f"  [display] {text[:200]}")
                elif "image/png" in data_keys:
                    sections.append("  [display] (chart/plot image)")
                else:
                    sections.append(f"  [display] ({', '.join(data_keys)})")

    result = "\n".join(sections)
    # Cap total output to prevent overwhelming small models
    if len(result) > 4000:
        result = result[:4000] + "\n\n... (notebook output truncated)"
    return result


def search_files(query: str, path: str, config: dict) -> str:
    """Search for a string across files in a directory using grep."""
    if not is_path_allowed(path, config):
        return f"Access denied: '{path}' is not within a registered folder."
    if not os.path.isdir(path):
        return f"Not a directory: '{path}'"
    try:
        result = subprocess.run(
            ["grep", "-r", "-n", "-l", "--include=*.py", "--include=*.md",
             "--include=*.txt", "--include=*.json", "--include=*.yaml",
             "--include=*.yml", "--include=*.toml", "--include=*.ipynb",
             query, path],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            files = result.stdout.strip().split("\n")[:20]
            return f"Found '{query}' in {len(files)} file(s):\n" + "\n".join(f"  {f}" for f in files)
        return f"No matches for '{query}' in {path}"
    except subprocess.TimeoutExpired:
        return "Search timed out."
    except Exception as e:
        return f"Search error: {e}"


def tree(path: str, config: dict, max_depth: int = 2) -> str:
    """Show directory tree up to max_depth."""
    if not is_path_allowed(path, config):
        return f"Access denied: '{path}' is not within a registered folder."
    if not os.path.isdir(path):
        return f"Not a directory: '{path}'"
    lines = [path]
    _tree_recurse(path, "", max_depth, 0, lines)
    if len(lines) > 100:
        lines = lines[:100]
        lines.append("... (truncated)")
    return "\n".join(lines)


def _tree_recurse(path: str, prefix: str, max_depth: int, depth: int, lines: list):
    if depth >= max_depth:
        return
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return
    # Skip hidden dirs and common noise
    entries = [e for e in entries if not e.startswith(".") and e not in
               ("__pycache__", "node_modules", ".git", "venv", ".venv")]
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            lines.append(f"{prefix}{connector}{entry}/")
            extension = "    " if is_last else "│   "
            _tree_recurse(full, prefix + extension, max_depth, depth + 1, lines)
        else:
            lines.append(f"{prefix}{connector}{entry}")


def find_notebooks(config: dict, keyword: str = "") -> str:
    """Find all .ipynb notebooks across registered folders, optionally filtered by keyword."""
    results = []
    for folder in config.get("registered_folders", []):
        for nb_path in Path(folder).rglob("*.ipynb"):
            if ".ipynb_checkpoints" in str(nb_path):
                continue
            name = str(nb_path)
            if keyword and keyword.lower() not in name.lower():
                continue
            size = nb_path.stat().st_size
            try:
                with open(nb_path) as f:
                    nb = json.load(f)
                n_cells = len(nb.get("cells", []))
                # Count cells with outputs
                n_with_output = sum(
                    1 for c in nb.get("cells", [])
                    if c.get("outputs") and len(c["outputs"]) > 0
                )
                results.append(f"  {name}  ({n_cells} cells, {n_with_output} with output)")
            except Exception:
                results.append(f"  {name}  (parse error)")
    if not results:
        return f"No notebooks found{' matching ' + repr(keyword) if keyword else ''}."
    return f"Found {len(results)} notebook(s):\n" + "\n".join(results)


# Tool definitions for Ollama's tool calling API
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Get Jarvis status: which folders are registered, how many files, what model is running.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. For .ipynb notebooks, this automatically parses and shows code cells with their outputs and metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "max_lines": {"type": "integer", "description": "Max lines to read (default 200)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_notebook",
            "description": "Parse a Jupyter .ipynb notebook and extract all code cells with their outputs, metrics, training logs, and errors. Use this to check the status of notebook runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the .ipynb notebook file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_notebooks",
            "description": "Find all Jupyter .ipynb notebooks across registered folders. Optionally filter by keyword. Shows cell counts and which have outputs (have been run).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Optional keyword to filter notebook names (e.g. 'v12', 'architecture')"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a string/pattern across files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search string"},
                    "path": {"type": "string", "description": "Directory to search in"}
                },
                "required": ["query", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tree",
            "description": "Show directory tree structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Root path for tree"},
                    "max_depth": {"type": "integer", "description": "Max depth (default 2)"}
                },
                "required": ["path"]
            }
        }
    }
]


def execute_tool(name: str, arguments: dict, config: dict) -> str:
    """Execute a tool by name with given arguments."""
    if name == "get_status":
        return get_status(config)
    elif name == "list_directory":
        return list_directory(arguments["path"], config)
    elif name == "read_file":
        return read_file(
            arguments["path"], config,
            max_lines=arguments.get("max_lines", 0)
        )
    elif name == "read_notebook":
        return read_notebook(arguments["path"], config)
    elif name == "find_notebooks":
        return find_notebooks(config, keyword=arguments.get("keyword", ""))
    elif name == "search_files":
        return search_files(arguments["query"], arguments["path"], config)
    elif name == "tree":
        return tree(
            arguments["path"], config,
            max_depth=arguments.get("max_depth", 2)
        )
    else:
        return f"Unknown tool: {name}"
