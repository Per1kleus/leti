"""
Meeting scheduling: creates a calendar event via CalDAV, optionally generates
a Zoom or Microsoft Teams meeting link, and can send a proper invite email
(with an .ics attachment) to participants.

Why CalDAV for the calendar instead of a vendor-specific API: it's the same
"open protocol + app password" shape already used for email (imaplib/smtplib)
rather than a full interactive OAuth2 consent flow, and it works against
Google Calendar, iCloud, Fastmail, Nextcloud, and most other providers
without needing a browser redirect this terminal/voice assistant has no easy
way to host.

Zoom uses Server-to-Server OAuth (an app-level credential, no per-user
consent needed - see Zoom App Marketplace). Teams uses Microsoft Graph
app-only auth (client credentials against an Azure AD app registration with
admin-consented OnlineMeetings.ReadWrite.All) creating the meeting under a
specific organizer mailbox. Both are "risky" tier since they create a real,
joinable meeting using your account.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from typing import Any, Dict, List, Optional

import httpx

from core.config_loader import get_settings
from tools.base import BaseTool, ToolParameter, ToolResult
from tools.email_client import _email_settings


def _parse_dt(value: str) -> datetime:
    """Accepts standard ISO 8601, including a trailing 'Z' (which python's
    fromisoformat only started accepting natively in 3.11)."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fmt_ical(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --- Zoom (Server-to-Server OAuth) --------------------------------------------

def _zoom_settings() -> Dict[str, str]:
    cfg = get_settings().get("zoom", {})
    if not all(cfg.get(k) for k in ("account_id", "client_id", "client_secret")):
        raise RuntimeError(
            "No [zoom] section in config/settings.yaml. Create a Server-to-Server OAuth "
            "app in the Zoom App Marketplace and add account_id, client_id, client_secret."
        )
    return cfg


async def _zoom_create_meeting(topic: str, start: datetime, duration_minutes: int) -> Dict[str, Any]:
    cfg = _zoom_settings()
    async with httpx.AsyncClient(timeout=15) as client:
        token_resp = await client.post(
            "https://zoom.us/oauth/token",
            params={"grant_type": "account_credentials", "account_id": cfg["account_id"]},
            auth=(cfg["client_id"], cfg["client_secret"]),
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]

        meeting_resp = await client.post(
            "https://api.zoom.us/v2/users/me/meetings",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "topic": topic,
                "type": 2,  # scheduled meeting
                "start_time": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration": duration_minutes,
                "timezone": "UTC",
                "settings": {"join_before_host": True, "waiting_room": False},
            },
        )
        meeting_resp.raise_for_status()
        data = meeting_resp.json()
        return {"join_url": data.get("join_url"), "meeting_id": data.get("id"), "password": data.get("password")}


# --- Microsoft Teams (Graph app-only auth) ------------------------------------

def _teams_settings() -> Dict[str, str]:
    cfg = get_settings().get("teams", {})
    if not all(cfg.get(k) for k in ("tenant_id", "client_id", "client_secret", "organizer_user_id")):
        raise RuntimeError(
            "No [teams] section in config/settings.yaml. Register an Azure AD app with "
            "admin-consented OnlineMeetings.ReadWrite.All and add tenant_id, client_id, "
            "client_secret, organizer_user_id (the mailbox that will own the meeting)."
        )
    return cfg


async def _teams_create_meeting(topic: str, start: datetime, end: datetime) -> Dict[str, Any]:
    cfg = _teams_settings()
    async with httpx.AsyncClient(timeout=15) as client:
        token_resp = await client.post(
            f"https://login.microsoftonline.com/{cfg['tenant_id']}/oauth2/v2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "scope": "https://graph.microsoft.com/.default",
            },
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]

        meeting_resp = await client.post(
            f"https://graph.microsoft.com/v1.0/users/{cfg['organizer_user_id']}/onlineMeetings",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "subject": topic,
                "startDateTime": start.astimezone(timezone.utc).isoformat(),
                "endDateTime": end.astimezone(timezone.utc).isoformat(),
            },
        )
        meeting_resp.raise_for_status()
        data = meeting_resp.json()
        return {"join_url": data.get("joinWebUrl"), "meeting_id": data.get("id")}


async def _create_video_meeting(platform: str, topic: str, start: datetime, end: datetime, duration_minutes: int) -> Optional[Dict[str, Any]]:
    if platform == "zoom":
        return await _zoom_create_meeting(topic, start, duration_minutes)
    if platform == "teams":
        return await _teams_create_meeting(topic, start, end)
    if platform == "none":
        return None
    raise ValueError(f"Unknown platform: {platform}")


# --- Calendar (CalDAV) ---------------------------------------------------------

def _calendar_settings() -> Dict[str, Any]:
    cfg = get_settings().get("calendar", {})
    if not all(cfg.get(k) for k in ("caldav_url", "username", "app_password")):
        raise RuntimeError(
            "No [calendar] section in config/settings.yaml. Add caldav_url, username, "
            "app_password (most providers, incl. Google/iCloud, need an app password for "
            "CalDAV once 2FA is on - see the commented calendar: block there)."
        )
    return cfg


def _escape_ics_text(text: str) -> str:
    """RFC 5545 TEXT escaping - applies to every free-text field (SUMMARY, LOCATION,
    DESCRIPTION), not just DESCRIPTION: backslash, semicolon, and comma are all
    value-grammar-significant characters and must be escaped wherever they appear,
    not only where a particular server's parser happens to be lenient about it."""
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _build_ics(
    uid: str, title: str, start: datetime, end: datetime,
    description: str, location: str, organizer_email: str, attendee_emails: List[str],
) -> str:
    """Builds a VCALENDAR/VEVENT by hand rather than depending on the icalendar library's
    object model, so the exact same text is used both for the CalDAV PUT and the .ics email
    attachment - one source of truth for what the meeting actually contains."""
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Leti//Meeting Scheduler//EN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_fmt_ical(datetime.now(timezone.utc))}",
        f"DTSTART:{_fmt_ical(start)}",
        f"DTEND:{_fmt_ical(end)}",
        f"SUMMARY:{_escape_ics_text(title)}",
    ]
    if location:
        lines.append(f"LOCATION:{_escape_ics_text(location)}")
    if description:
        lines.append(f"DESCRIPTION:{_escape_ics_text(description)}")
    if organizer_email:
        lines.append(f"ORGANIZER:mailto:{organizer_email}")
    for addr in attendee_emails:
        lines.append(f"ATTENDEE;RSVP=TRUE;CN={_escape_ics_text(addr)}:mailto:{addr}")
    lines += ["STATUS:CONFIRMED", "SEQUENCE:0", "END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines) + "\r\n"


def _save_to_caldav_blocking(ics_text: str, calendar_name: str = "") -> None:
    import caldav  # imported lazily so the whole tool module doesn't hard-fail if unused

    cfg = _calendar_settings()
    client = caldav.DAVClient(url=cfg["caldav_url"], username=cfg["username"], password=cfg["app_password"])
    principal = client.principal()
    calendars = principal.calendars()
    if not calendars:
        raise RuntimeError("No calendars found for this CalDAV account.")

    target = calendars[0]
    wanted = calendar_name or cfg.get("calendar_name", "")
    if wanted:
        for cal in calendars:
            if cal.name and cal.name.lower() == wanted.lower():
                target = cal
                break

    target.save_event(ical=ics_text)


async def _save_to_caldav(ics_text: str, calendar_name: str = "") -> None:
    """The caldav client is synchronous HTTP - several round trips against a remote
    server. Run on the event loop it stalls every other coroutine, including the GUI's
    websocket server."""
    await asyncio.get_running_loop().run_in_executor(
        None, _save_to_caldav_blocking, ics_text, calendar_name
    )


class ScheduleMeetingTool(BaseTool):
    name = "schedule_meeting"
    description = (
        "Schedule a meeting: creates a calendar event (via CalDAV) and, if a video platform is "
        "given, generates a real Zoom or Teams join link and embeds it in the event description. "
        "Does NOT email participants - use send_meeting_invite_email separately for that. "
        "Resolve any names to email addresses with resolve_contact before calling this."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="title", type="string", description="Meeting title."),
        ToolParameter(name="start_time", type="string", description="ISO 8601 start time, e.g. '2026-09-02T15:00:00-04:00'."),
        ToolParameter(name="duration_minutes", type="number", description="Duration in minutes."),
        ToolParameter(
            name="platform", type="string", enum=["zoom", "teams", "none"],
            description="Video platform to generate a join link for, or 'none' for an in-person/no-link event.",
        ),
        ToolParameter(name="participant_emails", type="array", items_type="string", required=False, description="Attendee email addresses."),
        ToolParameter(name="location", type="string", required=False, description="Physical location, if any."),
        ToolParameter(name="notes", type="string", required=False, description="Extra agenda/notes for the event description."),
    ]

    async def run(
        self, title: str, start_time: str, duration_minutes: int, platform: str,
        participant_emails: Optional[List[str]] = None, location: str = "", notes: str = "", **kwargs
    ) -> ToolResult:
        try:
            participant_emails = participant_emails or []
            start = _parse_dt(start_time)
            end = start + timedelta(minutes=duration_minutes)

            video = await _create_video_meeting(platform, title, start, end, duration_minutes)
            join_url = video["join_url"] if video else ""

            description_parts = []
            if notes:
                description_parts.append(notes)
            if join_url:
                description_parts.append(f"Join link ({platform}): {join_url}")
            description = "\n\n".join(description_parts)

            uid = f"{uuid.uuid4()}@leti"
            organizer_email = _email_settings().get("username", "") if _has_email_config() else ""
            ics_text = _build_ics(uid, title, start, end, description, location, organizer_email, participant_emails)
            await _save_to_caldav(ics_text)

            return ToolResult(success=True, output={
                "title": title,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "platform": platform,
                "join_url": join_url,
                "participants": participant_emails,
                "calendar_event_uid": uid,
                "ics": ics_text,
            })
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"{platform.title()} API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def _has_email_config() -> bool:
    try:
        _email_settings()
        return True
    except Exception:
        return False


class CreateVideoMeetingLinkTool(BaseTool):
    name = "create_video_meeting_link"
    description = "Generate a real, joinable Zoom or Teams meeting link without touching the calendar or email."
    parameters: List[ToolParameter] = [
        ToolParameter(name="topic", type="string", description="Meeting topic/title."),
        ToolParameter(name="start_time", type="string", description="ISO 8601 start time."),
        ToolParameter(name="duration_minutes", type="number", description="Duration in minutes."),
        ToolParameter(name="platform", type="string", enum=["zoom", "teams"], description="Which platform to use."),
    ]

    async def run(self, topic: str, start_time: str, duration_minutes: int, platform: str, **kwargs) -> ToolResult:
        try:
            start = _parse_dt(start_time)
            end = start + timedelta(minutes=duration_minutes)
            video = await _create_video_meeting(platform, topic, start, end, duration_minutes)
            return ToolResult(success=True, output=video)
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"{platform.title()} API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SendMeetingInviteEmailTool(BaseTool):
    name = "send_meeting_invite_email"
    description = (
        "Send a meeting invite email to participants, with the meeting details and join link in "
        "the body plus a proper .ics calendar attachment they can add with one click. Risky: "
        "requires confirmation before sending. If the event was already created with "
        "schedule_meeting, pass that call's calendar_event_uid so attendees' replies match "
        "the organizer's copy of the event."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="to", type="array", items_type="string", description="Recipient email addresses."),
        ToolParameter(
            name="calendar_event_uid", type="string", required=False,
            description=(
                "The calendar_event_uid returned by schedule_meeting for this same meeting. "
                "Omit only when no calendar event was created."
            ),
        ),
        ToolParameter(name="title", type="string", description="Meeting title."),
        ToolParameter(name="start_time", type="string", description="ISO 8601 start time."),
        ToolParameter(name="duration_minutes", type="number", description="Duration in minutes."),
        ToolParameter(name="join_url", type="string", required=False, description="Video meeting join link, if any."),
        ToolParameter(name="location", type="string", required=False, description="Physical location, if any."),
        ToolParameter(name="notes", type="string", required=False, description="Extra agenda/notes."),
    ]

    async def run(
        self, to: List[str], title: str, start_time: str, duration_minutes: int,
        join_url: str = "", location: str = "", notes: str = "",
        calendar_event_uid: str = "", **kwargs
    ) -> ToolResult:
        try:
            import smtplib

            cfg = _email_settings()
            start = _parse_dt(start_time)
            end = start + timedelta(minutes=duration_minutes)

            body_lines = [
                f"You're invited: {title}",
                "",
                f"When: {start.strftime('%A, %B %d, %Y at %I:%M %p %Z')} ({duration_minutes} min)",
            ]
            if location:
                body_lines.append(f"Where: {location}")
            if join_url:
                body_lines.append(f"Join link: {join_url}")
            if notes:
                body_lines += ["", notes]
            body_lines += ["", "A calendar invite (.ics) is attached - open it to add this to your calendar."]
            body = "\n".join(body_lines)

            # An iTIP REQUEST is matched to an existing event by UID. Minting a fresh
            # one here meant the attendee's RSVP carried a UID that matched nothing in
            # the organizer's calendar, so replies silently failed to attach to the
            # event schedule_meeting had just created.
            uid = calendar_event_uid or f"{uuid.uuid4()}@leti"
            ics_text = _build_ics(uid, title, start, end, notes, location, cfg.get("username", ""), to)

            msg = MIMEMultipart()
            msg["From"] = cfg["username"]
            msg["To"] = ", ".join(to)
            msg["Subject"] = f"Meeting invite: {title}"
            msg.attach(MIMEText(body, "plain"))

            ics_part = MIMEBase("text", "calendar", method="REQUEST", name="invite.ics")
            ics_part.set_payload(ics_text)
            encoders.encode_base64(ics_part)
            ics_part.add_header("Content-Disposition", "attachment", filename="invite.ics")
            msg.attach(ics_part)

            def _send() -> None:
                with smtplib.SMTP_SSL(cfg["smtp_host"], cfg.get("smtp_port", 465)) as server:
                    server.login(cfg["username"], cfg["app_password"])
                    server.sendmail(cfg["username"], to, msg.as_string())

            # Blocking SMTP over the network - keep it off the event loop.
            await asyncio.get_running_loop().run_in_executor(None, _send)

            return ToolResult(success=True, output={
                "sent_to": to,
                "calendar_event_uid": uid,
                "matched_existing_event": bool(calendar_event_uid),
                "summary": f"Invite sent to {', '.join(to)}.",
            })
        except Exception as e:
            return ToolResult(success=False, error=str(e))
