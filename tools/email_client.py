"""
Email: read/categorize inbox, send mail. Uses plain IMAP (read) and SMTP
(send) over SSL with an app password - no third-party mail API needed, so
it works with Gmail/Outlook/iCloud/etc. as long as the user enables an
app password (most providers require this over plain password auth).

Categorization is rule-based and local (no message content ever leaves
the machine): a scored mix of headers, sender patterns and keywords
sorts new mail into "crucial" / "financial" / "promotions" / "general",
plus an importance score so the most urgent things surface first.
"""
from __future__ import annotations

import email
import imaplib
import re
import smtplib
from dataclasses import dataclass, field
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

from core.config_loader import get_settings
from tools.base import BaseTool, ToolParameter, ToolResult

# --- Categorization heuristics -------------------------------------------------

FINANCIAL_KEYWORDS = [
    "invoice", "statement", "payment", "receipt", "balance due", "tax",
    "refund", "transaction", "billing", "autopay", "direct deposit",
    "wire transfer", "chargeback",
]
FINANCIAL_SENDER_HINTS = [
    "bank", "paypal", "stripe", "chase", "amex", "americanexpress", "wellsfargo",
    "visa", "mastercard", "venmo", "irs.gov", "coinbase", "binance", "robinhood",
    "fidelity", "vanguard", "schwab", "alpaca",
]
PROMO_KEYWORDS = [
    "% off", "sale", "discount", "deal", "limited time", "coupon", "clearance",
    "free shipping", "unsubscribe", "newsletter",
]
CRUCIAL_KEYWORDS = [
    "urgent", "action required", "security alert", "verify your account",
    "password reset", "suspicious sign-in", "account locked", "immediately",
    "final notice", "overdue",
]


@dataclass
class EmailSummary:
    uid: str
    subject: str
    sender: str
    date: str
    snippet: str
    category: str
    importance: int
    has_unsubscribe: bool = False


def _decode(value: Optional[str]) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    out = []
    for text, enc in parts:
        if isinstance(text, bytes):
            out.append(text.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(text)
    return "".join(out)


def _classify(subject: str, sender: str, snippet: str, is_vip: bool, has_unsubscribe: bool) -> tuple[str, int]:
    text = f"{subject} {snippet}".lower()
    sender_l = sender.lower()

    score = 0
    category = "general"

    if is_vip:
        score += 40

    if any(k in text for k in CRUCIAL_KEYWORDS):
        score += 45
        category = "crucial"

    if any(h in sender_l for h in FINANCIAL_SENDER_HINTS) or any(k in text for k in FINANCIAL_KEYWORDS):
        score += 30
        if category == "general":
            category = "financial"

    if has_unsubscribe or any(k in text for k in PROMO_KEYWORDS):
        score += 5  # promos rarely urgent even if scored elsewhere
        if category == "general":
            category = "promotions"

    score = max(0, min(100, score))
    return category, score


class _ImapSession:
    """Thin context manager around imaplib for a read-only peek at the inbox."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.conn: Optional[imaplib.IMAP4_SSL] = None

    def __enter__(self):
        self.conn = imaplib.IMAP4_SSL(self.settings["imap_host"], self.settings.get("imap_port", 993))
        self.conn.login(self.settings["username"], self.settings["app_password"])
        self.conn.select("INBOX", readonly=True)
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.conn:
                self.conn.logout()
        except Exception:
            pass


def _email_settings() -> Dict[str, Any]:
    settings = get_settings()
    if "email" not in settings:
        raise RuntimeError(
            "No [email] section in config/settings.yaml. Add imap_host, imap_port, smtp_host, "
            "smtp_port, username, app_password, and optional vip_senders before using email tools."
        )
    return settings["email"]


class ListNewEmailsTool(BaseTool):
    name = "list_new_emails"
    description = (
        "Fetch unread emails, categorize each as crucial / financial / promotions / general, "
        "score importance, and return them sorted most-important first. Read-only (peek mode - "
        "does not mark messages as read)."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="limit", type="number", description="Max emails to return.", required=False),
        ToolParameter(
            name="category", type="string", required=False,
            description="Filter to only this category.",
            enum=["crucial", "financial", "promotions", "general"],
        ),
    ]

    async def run(self, limit: int = 25, category: str = "", **kwargs) -> ToolResult:
        try:
            cfg = _email_settings()
            vip_senders = [s.lower() for s in cfg.get("vip_senders", [])]

            summaries: List[EmailSummary] = []
            with _ImapSession(cfg) as conn:
                status, data = conn.search(None, "UNSEEN")
                if status != "OK":
                    return ToolResult(success=False, error="IMAP search failed.")
                uids = data[0].split()[-200:]  # cap scan window
                for uid in reversed(uids):
                    status, msg_data = conn.fetch(uid, "(RFC822)")
                    if status != "OK" or not msg_data or not msg_data[0]:
                        continue
                    msg = email.message_from_bytes(msg_data[0][1])
                    subject = _decode(msg.get("Subject"))
                    sender = _decode(msg.get("From"))
                    date = msg.get("Date", "")
                    has_unsub = msg.get("List-Unsubscribe") is not None

                    snippet = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    snippet = part.get_payload(decode=True).decode(
                                        part.get_content_charset() or "utf-8", errors="replace"
                                    )[:300]
                                except Exception:
                                    pass
                                break
                    else:
                        try:
                            snippet = msg.get_payload(decode=True).decode(
                                msg.get_content_charset() or "utf-8", errors="replace"
                            )[:300]
                        except Exception:
                            snippet = ""

                    is_vip = any(v in sender.lower() for v in vip_senders)
                    cat, score = _classify(subject, sender, snippet, is_vip, has_unsub)
                    summaries.append(EmailSummary(
                        uid=uid.decode(), subject=subject, sender=sender, date=date,
                        snippet=snippet.strip().replace("\n", " ")[:200],
                        category=cat, importance=score, has_unsubscribe=has_unsub,
                    ))

                    if len(summaries) >= max(limit * 3, 50):
                        break  # enough raw candidates to filter/sort from

            if category:
                summaries = [s for s in summaries if s.category == category]
            summaries.sort(key=lambda s: s.importance, reverse=True)
            summaries = summaries[:limit]

            counts = {"crucial": 0, "financial": 0, "promotions": 0, "general": 0}
            for s in summaries:
                counts[s.category] = counts.get(s.category, 0) + 1

            return ToolResult(
                success=True,
                output={
                    "emails": [s.__dict__ for s in summaries],
                    "counts_shown": counts,
                    "summary": f"{len(summaries)} unread email(s) shown, sorted by importance.",
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SendEmailTool(BaseTool):
    name = "send_email"
    description = "Send an email via SMTP. Risky: requires confirmation before sending."
    parameters: List[ToolParameter] = [
        ToolParameter(name="to", type="string", description="Recipient email address."),
        ToolParameter(name="subject", type="string", description="Subject line."),
        ToolParameter(name="body", type="string", description="Plain-text body."),
        ToolParameter(name="cc", type="string", required=False, description="Comma-separated CC addresses."),
    ]

    async def run(self, to: str, subject: str, body: str, cc: str = "", **kwargs) -> ToolResult:
        try:
            cfg = _email_settings()
            msg = MIMEMultipart()
            msg["From"] = cfg["username"]
            msg["To"] = to
            if cc:
                msg["Cc"] = cc
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            recipients = [to] + [c.strip() for c in cc.split(",") if c.strip()]

            with smtplib.SMTP_SSL(cfg["smtp_host"], cfg.get("smtp_port", 465)) as server:
                server.login(cfg["username"], cfg["app_password"])
                server.sendmail(cfg["username"], recipients, msg.as_string())

            return ToolResult(success=True, output=f"Email sent to {to}.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
