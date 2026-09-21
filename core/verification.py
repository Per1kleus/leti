"""Did the thing Leti did actually happen?

A tool call that returns without raising has told you one fact: the call ran.
Whether the email reached anybody, whether the file on disk now holds what was
meant to be in it, whether the button that was clicked did anything - those are
different questions, and answering the first as if it were the second is how an
assistant ends up reporting work it did not do.

So every consequential action gets the same four-word vocabulary, which Coding
Mode has used since it existed and which now lives here so there is exactly one
of it:

    VERIFIED         something was checked and it holds
    NOT VERIFIED     it could not be confirmed - which is NOT the same as fine
    FAILED           it was checked and it does not hold
    NOT APPLICABLE   there is nothing here to verify

core/coding.py imports these and its own property checks stay where they are;
this module adds the other domains - files, mail, calendar, GitHub, the screen,
the business records, workflows - and the one summarise() everything shares.

Two rules shape every verifier below.

FREE, OR NOT AT ALL. A verifier may read local state that is already on this
machine: a file's bytes, a JSON store, a session object in memory. It must not
make a network call, take a screenshot, or ask the model anything. Verification
that costs a round trip on every turn is a tax on every turn, and it would make
the cheapest tools the slowest. When confirmation genuinely needs an external
look, the verifier says NOT VERIFIED and names the call that WOULD confirm it in
`to_confirm`, which is the honest answer and also the useful one.

THE RESULT PAYLOAD IS EVIDENCE ONLY WHEN THE OTHER SIDE PRODUCED IT. GitHub
returning a pull-request number is GitHub saying the pull request exists. A tool
returning {"ok": true} is Leti's own code saying it got that far. The first is
evidence; the second is the claim being checked.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("leti.verification")

VERIFIED = "VERIFIED"
NOT_VERIFIED = "NOT VERIFIED"
FAILED = "FAILED"
NOT_APPLICABLE = "NOT APPLICABLE"

RESULTS = (VERIFIED, NOT_VERIFIED, FAILED, NOT_APPLICABLE)

# Worst-first, so a summary can never come out better than its worst check.
_SEVERITY = {FAILED: 3, NOT_VERIFIED: 2, VERIFIED: 1, NOT_APPLICABLE: 0}


def check(property_name: str, result: str, detail: str, **extra: Any) -> Dict[str, Any]:
    """One check, in the shape everything here (and core/coding.py) produces."""
    if result not in RESULTS:
        raise ValueError(f"{result!r} is not one of {RESULTS}")
    out: Dict[str, Any] = {"property": property_name, "result": result, "detail": detail}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def summarise(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The overall answer, which is never better than its worst check."""
    results = [c.get("result") for c in checks]
    if FAILED in results:
        overall, why = FAILED, "at least one check failed"
    elif NOT_VERIFIED in results:
        overall, why = NOT_VERIFIED, "at least one property could not be confirmed"
    elif VERIFIED in results:
        overall, why = VERIFIED, "every applicable check passed"
    else:
        overall, why = NOT_APPLICABLE, "nothing could be checked"
    return {
        "overall": overall, "why": why, "checks": checks,
        "how_to_report": (
            "Report each property as it came back. NOT APPLICABLE means Leti could not "
            "check it, which is not the same as it being fine, and a test that was not "
            "run has not passed."),
    }


def worst(results: List[str]) -> str:
    """The most severe of several verdicts."""
    return max((r for r in results if r in _SEVERITY),
               key=lambda r: _SEVERITY[r], default=NOT_APPLICABLE)


# --------------------------------------------------------------------------- #
# Reading a tool's own result
#
# Tools return ToolResult, which the orchestrator turns into a dict or a string
# before anything else sees it. Neither shape is trusted to mean success here -
# these only answer "did the tool itself report a failure", which settles the
# FAILED case and nothing else.
# --------------------------------------------------------------------------- #

def _as_mapping(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {}


def tool_reported_failure(result: Any) -> Optional[str]:
    """The error a tool reported, or None. Not an opinion about success."""
    data = _as_mapping(result)
    if data.get("success") is False or data.get("ok") is False:
        return str(data.get("error") or data.get("detail") or "the tool reported a failure")
    if data.get("error"):
        return str(data["error"])
    return None


# --------------------------------------------------------------------------- #
# Domain verifiers
#
# Each takes (arguments, result) and returns one check. They are registered by
# tool name at the bottom; a tool with no verifier is NOT APPLICABLE, said out
# loud rather than assumed fine.
# --------------------------------------------------------------------------- #

def _path_arg(arguments: Dict[str, Any], *keys: str) -> Optional[Path]:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return Path(value).expanduser()
            except (OSError, ValueError):
                return None
    return None


def _verify_file_written(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    path = _path_arg(arguments, "path", "file_path", "destination_path")
    if path is None:
        return check("file written", NOT_VERIFIED,
                     "The call named no path, so there is nothing to look at.")
    try:
        if not path.exists():
            return check("file written", FAILED,
                         f"{path} does not exist after the write.", path=str(path))
        size = path.stat().st_size
    except OSError as e:
        return check("file written", NOT_VERIFIED,
                     f"{path} could not be read back ({e}).", path=str(path))
    content = arguments.get("content")
    if isinstance(content, str) and content:
        try:
            on_disk = path.read_text(errors="replace")
        except OSError as e:
            return check("file written", NOT_VERIFIED,
                         f"{path} exists ({size} bytes) but could not be read back ({e}).",
                         path=str(path), bytes=size)
        if content.strip() and content.strip() not in on_disk:
            return check("file written", FAILED,
                         f"{path} exists but does not contain what was written.",
                         path=str(path), bytes=size)
        return check("file written", VERIFIED,
                     f"{path} exists and holds what was written ({size} bytes).",
                     path=str(path), bytes=size)
    return check("file written", VERIFIED, f"{path} exists ({size} bytes).",
                 path=str(path), bytes=size,
                 limit="Existence and size were checked, not the contents.")


def _verify_file_deleted(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    path = _path_arg(arguments, "path", "file_path")
    if path is None:
        return check("file deleted", NOT_VERIFIED, "The call named no path.")
    try:
        gone = not path.exists()
    except OSError as e:
        return check("file deleted", NOT_VERIFIED, f"{path} could not be checked ({e}).")
    if gone:
        return check("file deleted", VERIFIED, f"{path} is no longer there.", path=str(path))
    return check("file deleted", FAILED, f"{path} still exists.", path=str(path))


def _verify_file_moved(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    source = _path_arg(arguments, "source_path", "source")
    destination = _path_arg(arguments, "destination_path", "destination")
    if source is None or destination is None:
        return check("file moved", NOT_VERIFIED, "The call did not name both ends of the move.")
    try:
        arrived, left = destination.exists(), not source.exists()
    except OSError as e:
        return check("file moved", NOT_VERIFIED, f"The move could not be checked ({e}).")
    if arrived and left:
        return check("file moved", VERIFIED, f"{destination} exists and {source} is gone.")
    if arrived:
        return check("file moved", NOT_VERIFIED,
                     f"{destination} exists but {source} is still there - this was a copy, "
                     "not a move.")
    return check("file moved", FAILED, f"{destination} does not exist.")


def _verify_email_sent(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    """Handed to the outgoing server is not received by the recipient.

    There is no free way to confirm delivery: it happens on somebody else's
    machine, minutes later, and the only signal Leti could ever get is a bounce
    that has not arrived yet. Saying VERIFIED here would be the exact lie this
    module exists to prevent.
    """
    to = arguments.get("to") or arguments.get("recipient") or "the recipient"
    return check(
        "email delivered", NOT_VERIFIED,
        f"The outgoing mail server accepted the message for {to}. Whether it was "
        "delivered, or landed in spam, happens elsewhere and Leti cannot see it.",
        accepted_by_server=True,
        to_confirm="list_new_emails later would show a bounce; nothing confirms delivery itself.")


def _verify_calendar_event(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    data = _as_mapping(result)
    # A CalDAV server that created the event hands back its URL/uid. That is the
    # server saying the event exists, which is evidence; our own "ok" is not.
    identifier = data.get("event_url") or data.get("uid") or data.get("event_id")
    if identifier:
        return check("meeting in the calendar", VERIFIED,
                     "The calendar server returned an identifier for the new event.",
                     event=str(identifier),
                     limit="The server accepted it; invitations to attendees are separate.")
    return check("meeting in the calendar", NOT_VERIFIED,
                 "The calendar server did not return an identifier for the event, so "
                 "Leti cannot say it is there.",
                 to_confirm="read the calendar for that day with business_calendar or "
                            "check_calendar_availability")


def _verify_github_write(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    data = _as_mapping(result)
    for key in ("html_url", "url", "number", "sha", "ref"):
        if data.get(key):
            return check("GitHub accepted it", VERIFIED,
                         f"GitHub returned {key}={data[key]} for this request.",
                         evidence={key: data[key]})
    nested = data.get("pull_request") or data.get("branch") or data.get("result")
    if isinstance(nested, dict):
        for key in ("html_url", "url", "number", "sha", "ref"):
            if nested.get(key):
                return check("GitHub accepted it", VERIFIED,
                             f"GitHub returned {key}={nested[key]} for this request.",
                             evidence={key: nested[key]})
    return check("GitHub accepted it", NOT_VERIFIED,
                 "GitHub did not return an identifier for what was created, so Leti "
                 "cannot confirm it exists.",
                 to_confirm="github with action 'pull_requests' or 'branches'")


def _verify_screen_action(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    """A click that landed is not a click that did anything.

    core/computer_use.py already tracks this per session - an action is recorded
    unverified until the screen is looked at again and the expectation holds. This
    reads that, rather than keeping a second count of it.
    """
    try:
        from core import computer_use
    except Exception as e:                                   # pragma: no cover
        return check("the screen changed", NOT_VERIFIED,
                     f"The computer-use session could not be read ({e}).")
    sessions = [s for s in computer_use.active_sessions()]
    if not sessions:
        return check("the screen changed", NOT_APPLICABLE,
                     "No computer-use session is open, so there is no expectation to "
                     "compare the screen against.")
    session = sessions[-1]
    pending = session.unverified_actions()
    if not pending:
        return check("the screen changed", VERIFIED,
                     "Every action in this session has been followed by a look at the "
                     "screen that matched what was expected.")
    return check("the screen changed", NOT_VERIFIED,
                 f"{len(pending)} action(s) in this session have not been confirmed by "
                 "looking at the screen afterwards.",
                 unconfirmed=[a.get("action") for a in pending][:5],
                 to_confirm="read_screen, then verify_screen with what you expected")


def _verify_record_written(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    """A business record is on this disk, so reading it back costs nothing."""
    data = _as_mapping(result)
    record = data.get("record") if isinstance(data.get("record"), dict) else {}
    record_id = data.get("id") or record.get("id")
    if not record_id:
        return check("record saved", NOT_VERIFIED,
                     "The tool did not return the id of the record it saved.",
                     to_confirm="list_business_data for that record type")
    try:
        from tools.business import load_records

        for records in load_records().values():
            if any(str(r.get("id")) == str(record_id) for r in records):
                return check("record saved", VERIFIED,
                             f"Record {record_id} is in the store.", record_id=str(record_id))
    except Exception as e:
        return check("record saved", NOT_VERIFIED,
                     f"The business store could not be read back ({e}).")
    return check("record saved", FAILED,
                 f"Record {record_id} is not in the store after the write.",
                 record_id=str(record_id))


def _verify_workflow_saved(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    data = _as_mapping(result)
    workflow_id = data.get("workflow_id") or data.get("id")
    if not workflow_id:
        return check("workflow saved", NOT_VERIFIED,
                     "The tool did not return a workflow id.",
                     to_confirm="list_workflows")
    try:
        from core import workflows

        found = workflows.get_workflow(str(workflow_id))
    except Exception as e:
        return check("workflow saved", NOT_VERIFIED,
                     f"The workflow store could not be read back ({e}).")
    if not found:
        return check("workflow saved", FAILED,
                     f"Workflow {workflow_id} is not in the store.", workflow_id=str(workflow_id))
    enabled = found.get("enabled")
    return check("workflow saved", VERIFIED,
                 f"Workflow '{found.get('name', workflow_id)}' is in the store"
                 + (" and enabled." if enabled else " (not enabled)."),
                 workflow_id=str(workflow_id), enabled=bool(enabled))


def _verify_scheduled_task(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    data = _as_mapping(result)
    job_id = data.get("task_id") or data.get("job_id") or data.get("id")
    if not job_id:
        return check("scheduled", NOT_VERIFIED,
                     "The tool did not return the id of the scheduled task.",
                     to_confirm="list_scheduled_tasks")
    try:
        from core import system_scheduler

        jobs = system_scheduler.load_jobs()
    except Exception as e:
        return check("scheduled", NOT_VERIFIED,
                     f"The scheduler store could not be read back ({e}).")
    match = next((j for j in jobs if str(j.get("id")) == str(job_id)), None)
    if not match:
        return check("scheduled", FAILED,
                     f"Scheduled task {job_id} is not in the scheduler.", job_id=str(job_id))
    return check("scheduled", VERIFIED,
                 f"'{match.get('name', job_id)}' is in the scheduler"
                 + (f", next run {match['next_run']}." if match.get("next_run") else "."),
                 job_id=str(job_id))


def _verify_command_ran(arguments: Dict[str, Any], result: Any) -> Dict[str, Any]:
    """Exit code 0 says the command ran, not that it did what was wanted."""
    data = _as_mapping(result)
    code = data.get("exit_code", data.get("returncode"))
    command = str(arguments.get("command") or arguments.get("cmd") or "")[:80]
    if code is None:
        return check("command did what was intended", NOT_VERIFIED,
                     "The command ran; no exit code came back, and nothing checked its effect.")
    if code != 0:
        return check("command did what was intended", FAILED,
                     f"`{command}` exited {code}. Read stderr before deciding what to do.",
                     exit_code=code)
    return check("command did what was intended", NOT_VERIFIED,
                 f"`{command}` exited 0, which means it ran without erroring - not that "
                 "the intended effect happened. Check the effect if it matters.",
                 exit_code=0)


# The map from tool name to how its real-world effect is confirmed. A tool that
# is not here has no verifier, which is reported as NOT APPLICABLE and never as
# success. Read-only tools deliberately stay out: there is nothing in the world
# for reading a file to have changed.
VERIFIERS: Dict[str, Callable[[Dict[str, Any], Any], Dict[str, Any]]] = {
    "write_file": _verify_file_written,
    "create_document": _verify_file_written,
    "delete_file": _verify_file_deleted,
    "move_file": _verify_file_moved,
    "send_email": _verify_email_sent,
    "send_meeting_invite_email": _verify_email_sent,
    "schedule_meeting": _verify_calendar_event,
    "create_video_meeting_link": _verify_calendar_event,
    "github": _verify_github_write,
    "mouse_click": _verify_screen_action,
    "keyboard_type": _verify_screen_action,
    "keyboard_hotkey": _verify_screen_action,
    "add_business_data": _verify_record_written,
    "update_business_data": _verify_record_written,
    "save_workflow": _verify_workflow_saved,
    "activate_workflow": _verify_workflow_saved,
    "create_scheduled_task": _verify_scheduled_task,
    "update_scheduled_task": _verify_scheduled_task,
    "run_shell_command": _verify_command_ran,
}

# Tools whose whole job is to read something. Naming them keeps "nothing to
# verify" apart from "nobody wrote a verifier yet" in the report.
READ_ONLY = frozenset({
    "read_file", "list_files", "read_screen", "web_search", "browser_read_page",
    "list_new_emails", "list_business_data", "business_dashboard", "system_report",
    "list_scheduled_tasks", "list_workflows", "view_user_profile", "code_map",
    "search_images", "get_weather", "inspect_project", "check_calendar_availability",
})


def verify(tool_name: str, arguments: Optional[Dict[str, Any]] = None,
           result: Any = None) -> Dict[str, Any]:
    """What is actually known about the effect of one tool call.

    Never raises: a verifier that blows up says NOT VERIFIED, because a
    verification layer that can fail a turn is worse than no verification layer.
    """
    arguments = arguments if isinstance(arguments, dict) else {}
    name = str(tool_name or "")

    failure = tool_reported_failure(result)
    if failure:
        return check(f"{name} succeeded", FAILED,
                     f"The tool reported a failure: {failure}", tool=name)

    if name in READ_ONLY:
        return check(f"{name} changed something", NOT_APPLICABLE,
                     "This reads; it does not change anything there would be an effect "
                     "to confirm.", tool=name)

    verifier = VERIFIERS.get(name)
    if verifier is None:
        return check(f"{name} had its intended effect", NOT_APPLICABLE,
                     "Leti has no way to check this tool's real-world effect. The call "
                     "ran; that is all that is known.", tool=name)
    try:
        out = verifier(arguments, result)
    except Exception as e:                                    # pragma: no cover - guard
        logger.debug(f"Verifier for {name} raised: {e}")
        return check(f"{name} had its intended effect", NOT_VERIFIED,
                     f"The check itself could not run ({e}).", tool=name)
    out.setdefault("tool", name)
    return out


# --------------------------------------------------------------------------- #
# What the model is told
# --------------------------------------------------------------------------- #

def note_for(check_result: Dict[str, Any]) -> str:
    """One line to append to a tool result, or nothing.

    Deliberately silent for VERIFIED and NOT APPLICABLE: a note on every call
    would be tokens spent telling the model things it can already see. The lines
    that matter are the ones that stop a claim being made.
    """
    verdict = check_result.get("result")
    if verdict == FAILED:
        return (f"[{FAILED}] {check_result.get('detail', '')} Do not report this as done.")
    if verdict == NOT_VERIFIED:
        extra = check_result.get("to_confirm")
        return (f"[{NOT_VERIFIED}] {check_result.get('detail', '')}"
                + (f" To confirm: {extra}." if extra else "")
                + " Say what is confirmed and what is not; do not report it as confirmed.")
    return ""
