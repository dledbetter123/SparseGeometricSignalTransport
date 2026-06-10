"""Email handling: IMAP listener for incoming mail, SMTP for replies."""

import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from datetime import datetime
import logging

logger = logging.getLogger("jarvis")

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def decode_header_value(value):
    """Decode an email header that may be encoded."""
    if value is None:
        return ""
    decoded_parts = decode_header(value)
    result: list[str] = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result)


def get_email_body(msg) -> str:
    """Extract plain text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def fetch_unread_emails(gmail_user: str, gmail_password: str, authorized_senders: list) -> list:
    """Fetch unread emails from authorized senders. Returns list of (uid, sender, subject, body)."""
    results = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(gmail_user, gmail_password)
        mail.select("INBOX")

        status, data = mail.uid("search", "UNSEEN")
        if status != "OK" or not data[0]:
            mail.logout()
            return results

        # Decode the byte strings to regular strings, since IMAP4.uid expects strings
        uids = [u.decode("utf-8") for u in data[0].split()]
        for uid in uids:
            status, msg_data = mail.uid("fetch", uid, "(RFC822)")
            if status != "OK":
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            sender = decode_header_value(msg.get("From", ""))
            subject = decode_header_value(msg.get("Subject", ""))
            body = get_email_body(msg)

            # Extract just the email address from "Name <email>" format
            sender_email = sender
            if "<" in sender and ">" in sender:
                sender_email = sender.split("<")[1].split(">")[0]

            # Check if sender is authorized
            if sender_email.lower() not in [s.lower() for s in authorized_senders]:
                logger.info(f"Ignoring email from unauthorized sender: {sender_email}")
                # Still mark as read so we don't keep checking it
                mail.uid("store", uid, "+FLAGS", "\\Seen")
                continue

            results.append({
                "uid": uid,
                "sender": sender,
                "sender_email": sender_email,
                "subject": subject,
                "body": body.strip(),
                "message_id": msg.get("Message-ID", ""),
            })

        mail.logout()
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")

    return results


def mark_as_read(gmail_user: str, gmail_password: str, uid: str):
    """Mark a specific email as read."""
    if isinstance(uid, bytes):
        uid = uid.decode("utf-8")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(gmail_user, gmail_password)
        mail.select("INBOX")
        mail.uid("store", uid, "+FLAGS", "\\Seen")
        mail.logout()
    except Exception as e:
        logger.error(f"Error marking email as read: {e}")


def send_reply(gmail_user: str, gmail_password: str, to_email: str,
               subject: str, body: str, in_reply_to: str = ""):
    """Send a reply email."""
    try:
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to_email
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_body = f"{body}\n\n---\nJarvis | {timestamp}"

        msg.attach(MIMEText(full_body, "plain"))

        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()

        logger.info(f"Reply sent to {to_email}: {subject}")
    except Exception as e:
        logger.error(f"Error sending reply: {e}")
