"""CLI entry point for Jarvis."""

import argparse
import logging
import sys
import os

from jarvis.config import (
    load_config, save_config, register_folder, unregister_folder,
    LOG_PATH, JARVIS_HOME,
)
from jarvis.agent import run_agent
from jarvis.service import run_service


def setup_logging(verbose=False):
    JARVIS_HOME.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )


def cmd_configure(args):
    """Interactive configuration."""
    config = load_config()

    gmail_user = input(f"Gmail address [{config.get('gmail_user', '')}]: ").strip()
    if gmail_user:
        config["gmail_user"] = gmail_user

    gmail_pass = input("Gmail app password [hidden]: ").strip()
    if gmail_pass:
        config["gmail_app_password"] = gmail_pass

    model = input(f"Ollama model [{config.get('model', 'qwen3.5:0.8b')}]: ").strip()
    if model:
        config["model"] = model

    save_config(config)
    print(f"Configuration saved to {JARVIS_HOME / 'config.json'}")


def cmd_register(args):
    """Register a folder."""
    path = args.path or os.getcwd()
    print(register_folder(path))


def cmd_unregister(args):
    """Unregister a folder."""
    print(unregister_folder(args.path))


def cmd_list(args):
    """List registered folders."""
    config = load_config()
    folders = config.get("registered_folders", [])
    if not folders:
        print("No folders registered.")
        return
    print(f"Registered folders ({len(folders)}):")
    for f in folders:
        exists = "OK" if os.path.isdir(f) else "NOT FOUND"
        print(f"  [{exists}] {f}")
    print(f"\nModel: {config.get('model', 'not set')}")
    print(f"Gmail: {config.get('gmail_user', 'not set')}")


def cmd_start(args):
    """Start the email polling service."""
    setup_logging(verbose=args.verbose)
    run_service()


def cmd_ask(args):
    """Ask Jarvis a question directly (no email)."""
    setup_logging(verbose=False)
    logging.getLogger("jarvis").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    question = " ".join(args.question)
    if not question:
        print("Usage: jarvis ask <question>")
        return
    response = run_agent(question)
    print(response)

    from jarvis.service import log_message
    log_message(sender="cli", subject="", body=question, response=response, source="cli")


def cmd_status(args):
    """Show Jarvis status."""
    from jarvis.tools import get_status
    config = load_config()
    print(get_status(config))


def cmd_log(args):
    """Show the message log."""
    from jarvis.config import MESSAGES_PATH
    import json

    if not MESSAGES_PATH.exists():
        print("No messages logged yet.")
        return

    with open(MESSAGES_PATH) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print("No messages logged yet.")
        return

    # Apply filters
    if args.source:
        entries = [e for e in entries if e.get("source") == args.source]

    # Show last N
    count = args.last or 10
    entries = entries[-count:]

    for entry in entries:
        ts = entry.get("timestamp", "?")[:19]
        src = entry.get("source", "?")
        sender = entry.get("sender", "?")
        subject = entry.get("subject", "")
        msg = entry.get("message", "")
        resp = entry.get("response", "")

        print(f"{'='*60}")
        print(f"[{ts}] {src} | from: {sender}")
        if subject:
            print(f"Subject: {subject}")
        print(f"\n> {msg[:300]}")
        if len(msg) > 300:
            print(f"  ... ({len(msg)} chars total)")
        print(f"\n{resp[:500]}")
        if len(resp) > 500:
            print(f"  ... ({len(resp)} chars total)")
        print()

    print(f"Showing {len(entries)} message(s). Log: {MESSAGES_PATH}")


def cmd_install_service(args):
    """Install as a macOS launchd service."""
    python_path = sys.executable
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jarvis.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>jarvis.cli</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{JARVIS_HOME}/jarvis_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{JARVIS_HOME}/jarvis_stderr.log</string>
    <key>WorkingDirectory</key>
    <string>{os.path.expanduser('~')}</string>
</dict>
</plist>"""

    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.jarvis.agent.plist")
    with open(plist_path, "w") as f:
        f.write(plist_content)

    print(f"LaunchAgent plist written to: {plist_path}")
    print("\nTo start the service:")
    print(f"  launchctl load {plist_path}")
    print("\nTo stop the service:")
    print(f"  launchctl unload {plist_path}")
    print("\nTo check status:")
    print(f"  launchctl list | grep jarvis")


def main():
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Jarvis - Local AI agent accessible via email",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # configure
    subparsers.add_parser("configure", help="Configure Jarvis credentials and settings")

    # register
    p_reg = subparsers.add_parser("register", help="Register a folder for Jarvis to monitor")
    p_reg.add_argument("path", nargs="?", default=None, help="Folder path (default: current directory)")

    # unregister
    p_unreg = subparsers.add_parser("unregister", help="Unregister a folder")
    p_unreg.add_argument("path", help="Folder path to unregister")

    # list
    subparsers.add_parser("list", help="List registered folders")

    # start
    p_start = subparsers.add_parser("start", help="Start the email polling service")
    p_start.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask Jarvis a question directly")
    p_ask.add_argument("question", nargs="+", help="Your question")

    # status
    subparsers.add_parser("status", help="Show Jarvis status")

    # log
    p_log = subparsers.add_parser("log", help="Show message log")
    p_log.add_argument("-n", "--last", type=int, default=10, help="Show last N messages (default 10)")
    p_log.add_argument("--source", choices=["email", "cli"], help="Filter by source")

    # install-service
    subparsers.add_parser("install-service", help="Install as macOS launchd service")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "configure": cmd_configure,
        "register": cmd_register,
        "unregister": cmd_unregister,
        "list": cmd_list,
        "start": cmd_start,
        "ask": cmd_ask,
        "status": cmd_status,
        "log": cmd_log,
        "install-service": cmd_install_service,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
