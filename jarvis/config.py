"""Configuration management for Jarvis. Stores state in ~/.jarvis/config.json."""

import json
import os
from pathlib import Path
from typing import Any

JARVIS_HOME = Path.home() / ".jarvis"
CONFIG_PATH = JARVIS_HOME / "config.json"
MESSAGES_PATH = JARVIS_HOME / "messages.jsonl"
LOG_PATH = JARVIS_HOME / "jarvis.log"

DEFAULT_CONFIG = {
    "model": "qwen3.5:0.8b",
    "poll_interval_seconds": 60,
    "max_tool_iterations": 5,
    "max_file_lines": 200,
    "authorized_senders": ["dledbetter456@gmail.com", "david@holonomy-ai.com"],
    "registered_folders": [],
    "gmail_user": "",
    "gmail_app_password": "",
}


def ensure_home():
    JARVIS_HOME.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    ensure_home()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            saved_data = json.load(f)
        if not isinstance(saved_data, dict):
            saved_data = {}
        saved: dict[str, Any] = saved_data
        # Merge with defaults — for lists, union default + saved entries
        merged: dict[str, Any] = {}
        for key, default_val in DEFAULT_CONFIG.items():
            if key not in saved:
                merged[key] = default_val
            elif isinstance(default_val, list):
                # Union: keep all saved entries + any new defaults
                combined = list(saved[key])
                for item in default_val:
                    if item not in combined:
                        combined.append(item)
                merged[key] = combined
            else:
                merged[key] = saved[key]
        # Preserve any extra keys from saved config
        for key in saved:
            if key not in merged:
                merged[key] = saved[key]
        return merged
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    ensure_home()
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def register_folder(folder_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(folder_path))
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a directory."
    config = load_config()
    if path in config["registered_folders"]:
        return f"Already registered: {path}"
    config["registered_folders"].append(path)
    save_config(config)
    return f"Registered: {path}"


def unregister_folder(folder_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(folder_path))
    config = load_config()
    if path in config["registered_folders"]:
        config["registered_folders"].remove(path)
        save_config(config)
        return f"Unregistered: {path}"
    return f"Not registered: {path}"


def is_path_allowed(path: str, config: dict) -> bool:
    """Check that a path falls within a registered folder."""
    abs_path = os.path.abspath(path)
    for folder in config["registered_folders"]:
        if abs_path.startswith(folder):
            return True
    return False
