"""Main service loop: polls email, runs agent, sends replies."""

import json
import time
import logging
import signal
import sys
from datetime import datetime
from jarvis.config import load_config, MESSAGES_PATH, ensure_home
from jarvis.email_handler import fetch_unread_emails, mark_as_read, send_reply
from jarvis.agent import run_agent

logger = logging.getLogger("jarvis")


def log_message(sender: str, subject: str, body: str, response: str, source: str = "email"):
    """Append a conversation record to messages.jsonl."""
    ensure_home()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "sender": sender,
        "subject": subject,
        "message": body,
        "response": response,
    }
    with open(MESSAGES_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Shutdown signal received.")
    _running = False


def process_email(email_data: dict, config: dict):
    """Process a single email: run agent, send reply."""
    sender = email_data["sender_email"]
    subject = email_data["subject"]
    body = email_data["body"]

    # Build the user message from subject + body
    user_message = body
    if subject and subject.lower() not in ("(no subject)", ""):
        user_message = f"Subject: {subject}\n\n{body}"

    logger.info(f"Processing email from {sender}: {subject}")

    # Run the agent
    try:
        response = run_agent(user_message, config)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        response = f"Jarvis encountered an error processing your request: {e}"

    # Log the conversation
    log_message(sender=sender, subject=subject, body=body, response=response, source="email")

    # Send reply
    send_reply(
        gmail_user=config["gmail_user"],
        gmail_password=config["gmail_app_password"],
        to_email=sender,
        subject=subject,
        body=response,
        in_reply_to=email_data.get("message_id", ""),
    )

    # Mark original as read
    mark_as_read(
        gmail_user=config["gmail_user"],
        gmail_password=config["gmail_app_password"],
        uid=email_data["uid"],
    )


def run_service():
    """Main service loop."""
    global _running

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    config = load_config()

    if not config.get("gmail_user") or not config.get("gmail_app_password"):
        logger.error("Gmail credentials not configured. Run: jarvis configure")
        sys.exit(1)

    poll_interval = config.get("poll_interval_seconds", 60)

    logger.info(f"Jarvis service started. Polling every {poll_interval}s.")
    logger.info(f"Model: {config.get('model', 'unknown')}")
    logger.info(f"Folders: {config.get('registered_folders', [])}")

    while _running:
        try:
            # Reload config each cycle in case it changed
            config = load_config()

            emails = fetch_unread_emails(
                gmail_user=config["gmail_user"],
                gmail_password=config["gmail_app_password"],
                authorized_senders=config.get("authorized_senders", []),
            )

            for email_data in emails:
                if not _running:
                    break
                process_email(email_data, config)

        except Exception as e:
            logger.error(f"Service loop error: {e}")

        # Sleep in small increments so we can respond to shutdown signals
        for _ in range(poll_interval):
            if not _running:
                break
            time.sleep(1)

    logger.info("Jarvis service stopped.")
