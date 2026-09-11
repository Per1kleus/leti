"""Scheduled tasks: recurring work that runs Leti's own capabilities.

A task is a natural-language instruction plus a schedule. When it fires, the
instruction goes through the orchestrator exactly as if the user had typed it -
which is what makes "every Monday, research 20 potential clients and write a
report" work without the scheduler knowing anything about research, data
analysis or reports. It doesn't need a per-capability task type, and adding a
capability elsewhere makes it schedulable here for free.

Scheduled meetings work the same way and are not a separate feature: a task
whose instruction is "book the Monday standup on Zoom" calls the existing
schedule_meeting and create_video_meeting_link tools when it fires.

Three things in this codebase have "schedule" in the name and none of them is a
copy of another. This file is the scheduler: it owns the task list, when each
task is due, and what happened when it ran. core/system_scheduler.py owns
nothing - it registers one recurring command with cron, launchd or schtasks so
the OS wakes Leti up to ask this file what is due, even when Leti is closed.
core/watches.py is a consumer: it keeps a single row in this file's store and
evaluates every watch when that row fires, rather than scheduling one row each.

What the scheduler owns is the part the orchestrator can't supply: when things
run, whether they succeeded, what they produced, and what to do when they fail.
Every run is recorded with its output and duration, failures are retried with
backoff, and a task that keeps failing is disabled rather than left to fail
forever in silence - with the user told either way.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.atomic_write import atomic_write_json
from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.scheduler")

SCHEDULE_TYPES = ("once", "interval", "daily", "weekly", "monthly")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# How many runs to keep per task. Enough to see a pattern of failures; not so
# many that the store grows without bound.
MAX_HISTORY_PER_TASK = 50
DEFAULT_MAX_RETRIES = 2
# A task whose instruction runs long shouldn't block the scheduler forever.
DEFAULT_TASK_TIMEOUT = 900


def _lock_path() -> Path:
    return resolve_path("./data/scheduler.lock")


@contextmanager
def scheduler_lock():
    """Hold the exclusive right to run due tasks, or yield False immediately.

    Once the OS scheduler is installed there are two things that want to run due
    tasks: the loop inside a running Leti, and the process cron/launchd/Task
    Scheduler starts. Without this they would both pick up the same due task and
    run it twice - sending the same email twice, writing the same report twice.

    An OS file lock rather than a flag in the task file, because a flag outlives
    the process that set it: if Leti is killed mid-run the kernel releases this
    automatically, while a stale flag would block every future run until someone
    noticed and cleared it.
    """
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _store_path() -> Path:
    cfg = get_settings().get("scheduler", {})
    path = resolve_path(cfg.get("file_path", "./data/scheduled_tasks.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_tasks() -> List[Dict[str, Any]]:
    import json

    path = _store_path()
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def save_tasks(tasks: List[Dict[str, Any]]) -> None:
    atomic_write_json(_store_path(), tasks)


def _parse_time_of_day(value: str) -> tuple:
    """'09:30' -> (9, 30). Defaults to 09:00 rather than midnight, since a daily
    task the user didn't give a time to almost always means 'in the morning'."""
    try:
        hour, _, minute = str(value or "09:00").partition(":")
        return max(0, min(int(hour), 23)), max(0, min(int(minute or 0), 59))
    except ValueError:
        return 9, 0


def compute_next_run(task: Dict[str, Any], after: Optional[float] = None) -> Optional[float]:
    """When this task should next run, as a timestamp, or None if never again.

    Local time throughout: "every Monday at 9" means the user's Monday morning,
    not UTC's.
    """
    now = datetime.fromtimestamp(after if after is not None else time.time())
    kind = task.get("schedule_type")
    hour, minute = _parse_time_of_day(task.get("at", "09:00"))

    if kind == "once":
        when = task.get("run_at")
        if not when:
            return None
        # A one-off that has already run doesn't run again.
        return None if task.get("last_run_at") else float(when)

    if kind == "interval":
        minutes = max(1.0, float(task.get("every_minutes", 60)))
        return (now + timedelta(minutes=minutes)).timestamp()

    if kind == "daily":
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate.timestamp()

    if kind == "weekly":
        wanted = [d.lower() for d in (task.get("weekdays") or ["monday"])]
        indices = sorted({WEEKDAYS.index(d) for d in wanted if d in WEEKDAYS}) or [0]
        for ahead in range(0, 8):
            candidate = (now + timedelta(days=ahead)).replace(
                hour=hour, minute=minute, second=0, microsecond=0)
            if candidate.weekday() in indices and candidate > now:
                return candidate.timestamp()
        return None

    if kind == "monthly":
        day = max(1, min(int(task.get("day_of_month", 1)), 28))
        # Capped at 28 deliberately: "the 31st" silently skips February and the
        # short months, which is a worse surprise than running a few days early.
        candidate = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            month = candidate.month + 1
            year = candidate.year + (month > 12)
            candidate = candidate.replace(year=year, month=1 if month > 12 else month)
        return candidate.timestamp()

    return None


def describe_schedule(task: Dict[str, Any]) -> str:
    kind = task.get("schedule_type")
    at = task.get("at", "09:00")
    if kind == "once":
        return f"once at {datetime.fromtimestamp(task['run_at']):%Y-%m-%d %H:%M}" if task.get("run_at") else "once"
    if kind == "interval":
        return f"every {task.get('every_minutes', 60)} minutes"
    if kind == "daily":
        return f"daily at {at}"
    if kind == "weekly":
        days = ", ".join(d.capitalize() for d in (task.get("weekdays") or ["monday"]))
        return f"every {days} at {at}"
    if kind == "monthly":
        return f"monthly on day {task.get('day_of_month', 1)} at {at}"
    return "unscheduled"


def _public(task: Dict[str, Any]) -> Dict[str, Any]:
    """A task as the model should see it - timestamps as readable dates, and the
    last few runs rather than the whole history."""
    history = task.get("history", [])
    return {
        "id": task["id"],
        "name": task.get("name"),
        "instruction": task.get("instruction"),
        "schedule": describe_schedule(task),
        "project": task.get("project"),
        "enabled": task.get("enabled", True),
        "next_run": (datetime.fromtimestamp(task["next_run"]).strftime("%Y-%m-%d %H:%M")
                     if task.get("next_run") else None),
        "last_run": (datetime.fromtimestamp(task["last_run_at"]).strftime("%Y-%m-%d %H:%M")
                     if task.get("last_run_at") else None),
        "last_status": task.get("last_status"),
        "consecutive_failures": task.get("consecutive_failures", 0),
        "disabled_reason": task.get("disabled_reason"),
        "runs": len(history),
        "recent_history": history[-5:],
    }


class SchedulerRunner:
    """Checks for due tasks and runs them through the orchestrator.

    Started by main.py once the orchestrator exists. One task at a time: the
    orchestrator already serialises turns, and running several scheduled tasks at
    once would interleave them with whatever the user is doing.
    """

    def __init__(self, orchestrator, notify: Optional[Callable[[str], Any]] = None,
                 check_interval: float = 30.0):
        self.orchestrator = orchestrator
        self.notify = notify
        self.check_interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.run_due_tasks()
            except asyncio.CancelledError:
                raise
            except Exception:
                # The scheduler outliving a bad task is the whole point of it.
                logger.exception("Scheduler loop error - continuing")
            await asyncio.sleep(self.check_interval)

    async def run_due_tasks(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Run everything that's due, if no other process is already doing it.

        A task overdue because the machine was off runs once when it next gets the
        chance, not once per missed occurrence - next_run only advances after a
        run, so a week of downtime doesn't produce seven reports.
        """
        now = now if now is not None else time.time()
        with scheduler_lock() as acquired:
            if not acquired:
                logger.debug("Another process is running due tasks; skipping this check.")
                return []
            due = [t for t in load_tasks()
                   if t.get("enabled", True) and t.get("next_run") and t["next_run"] <= now]
            return [await self.execute(task["id"]) for task in due]

    async def execute(self, task_id: str, manual: bool = False) -> Dict[str, Any]:
        """Run one task, record the outcome, schedule the next run.

        Reloads the task from the store on each attempt: a run can take minutes,
        and the user may have edited or disabled it meanwhile.
        """
        tasks = load_tasks()
        task = next((t for t in tasks if t["id"] == task_id), None)
        if task is None:
            return {"task_id": task_id, "status": "missing"}

        max_retries = int(task.get("max_retries", DEFAULT_MAX_RETRIES))
        timeout = float(task.get("timeout_seconds", DEFAULT_TASK_TIMEOUT))
        started = time.time()
        attempts: List[Dict[str, Any]] = []
        answer, status, error = None, "failed", None

        for attempt in range(max_retries + 1):
            try:
                instruction = task["instruction"]
                if task.get("project"):
                    # Say which project this belongs to in the instruction itself,
                    # so the orchestrator opens it and the work lands in the right
                    # folder - rather than the scheduler reaching into project state.
                    instruction = (
                        f"[Scheduled task '{task['name']}' for project "
                        f"'{task['project']}'. Open that project first.]\n{instruction}"
                    )
                else:
                    instruction = f"[Scheduled task '{task['name']}']\n{instruction}"

                answer = await asyncio.wait_for(
                    self.orchestrator.handle_user_input(instruction), timeout=timeout
                )
                status, error = "succeeded", None
                attempts.append({"attempt": attempt + 1, "status": "succeeded"})
                break
            except asyncio.TimeoutError:
                error = f"timed out after {timeout}s"
                attempts.append({"attempt": attempt + 1, "status": "timeout"})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                error = str(e)
                attempts.append({"attempt": attempt + 1, "status": "error", "error": str(e)})

            if attempt < max_retries:
                # Backoff, so a transient failure (a service down, a rate limit)
                # gets a real chance to clear rather than three failures in a row.
                await asyncio.sleep(min(30, 5 * (attempt + 1)))

        tasks = load_tasks()
        task = next((t for t in tasks if t["id"] == task_id), task)
        record = {
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - started, 1),
            "status": status,
            "manual": manual,
            "attempts": attempts,
            "output": (answer or "")[:4000] if status == "succeeded" else None,
            "error": error,
        }
        task.setdefault("history", []).append(record)
        task["history"] = task["history"][-MAX_HISTORY_PER_TASK:]
        task["last_run_at"] = started
        task["last_status"] = status
        task["consecutive_failures"] = 0 if status == "succeeded" else task.get("consecutive_failures", 0) + 1

        if not manual:
            task["next_run"] = compute_next_run(task)

        # A task failing every time is a broken task, and leaving it to fail
        # forever is how a scheduler becomes noise the user stops reading.
        failure_limit = int(get_settings().get("scheduler", {}).get("disable_after_failures", 5))
        if task["consecutive_failures"] >= failure_limit:
            task["enabled"] = False
            task["disabled_reason"] = (
                f"Disabled automatically after {task['consecutive_failures']} consecutive "
                f"failures. Last error: {error}"
            )

        for index, existing in enumerate(tasks):
            if existing["id"] == task_id:
                tasks[index] = task
                break
        save_tasks(tasks)

        if status != "succeeded" and self.notify:
            message = f"Scheduled task '{task['name']}' failed: {error}"
            if not task.get("enabled", True):
                message += " It has been disabled after repeated failures."
            try:
                result = self.notify(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Failed to deliver a scheduler notification")

        return {"task_id": task_id, "name": task["name"], "status": status,
                "error": error, "duration_seconds": record["duration_seconds"]}


# --- Tools -----------------------------------------------------------------------

_RUNNER: Optional[SchedulerRunner] = None


def set_runner(runner: SchedulerRunner) -> None:
    """main.py registers the live runner so run_scheduled_task_now can use it."""
    global _RUNNER
    _RUNNER = runner


class CreateScheduledTaskTool(BaseTool):
    name = "create_scheduled_task"
    description = (
        "Schedule work to happen automatically. The instruction is written the way the user "
        "would say it, and runs through your full set of tools when it fires - so "
        "'research 20 potential clients in Patras and write a report' will search, read "
        "pages, and write the report, with no special setup.\n"
        "Schedules: once (run_at), interval (every_minutes), daily (at), weekly (weekdays + "
        "at), monthly (day_of_month + at). Attach a project so the work lands in its folder.\n"
        "Write the instruction so it makes sense with no conversation around it - it runs on "
        "its own, weeks later, with none of this context."
    )
    parameters = [
        ToolParameter(name="name", type="string", description="Short name, e.g. 'Monday client research'."),
        ToolParameter(name="instruction", type="string",
                      description="What to do when it fires, written as a standalone request."),
        ToolParameter(name="schedule_type", type="string", enum=list(SCHEDULE_TYPES),
                      description="once, interval, daily, weekly, or monthly."),
        ToolParameter(name="at", type="string", required=False,
                      description="Time of day as HH:MM for daily/weekly/monthly (default 09:00)."),
        ToolParameter(name="weekdays", type="array", items_type="string", required=False,
                      description="For weekly: ['monday','thursday']."),
        ToolParameter(name="day_of_month", type="number", required=False,
                      description="For monthly: 1-28."),
        ToolParameter(name="every_minutes", type="number", required=False,
                      description="For interval: how many minutes between runs."),
        ToolParameter(name="run_at", type="string", required=False,
                      description="For once: ISO datetime, e.g. '2026-03-01T09:00'."),
        ToolParameter(name="project", type="string", required=False,
                      description="Project this belongs to, so output lands in its folder."),
        ToolParameter(name="max_retries", type="number", required=False,
                      description="Retries on failure before giving up for this run (default 2)."),
    ]

    async def run(self, name: str, instruction: str, schedule_type: str, at: str = "",
                  weekdays: Optional[List[str]] = None, day_of_month: int = 1,
                  every_minutes: float = 60, run_at: str = "", project: str = "",
                  max_retries: int = DEFAULT_MAX_RETRIES, **kwargs) -> ToolResult:
        if schedule_type not in SCHEDULE_TYPES:
            return ToolResult(success=False, error=f"schedule_type must be one of {SCHEDULE_TYPES}.")
        if not instruction.strip():
            return ToolResult(success=False, error="A scheduled task needs an instruction.")

        if weekdays:
            unknown = [d for d in weekdays if str(d).lower() not in WEEKDAYS]
            if unknown:
                return ToolResult(success=False, error=f"Unknown weekday(s): {unknown}.")

        task: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:8],
            "name": name or instruction[:40],
            "instruction": instruction,
            "schedule_type": schedule_type,
            "at": at or "09:00",
            "weekdays": [str(d).lower() for d in (weekdays or [])] or None,
            "day_of_month": int(day_of_month),
            "every_minutes": float(every_minutes),
            "project": project or None,
            "enabled": True,
            "max_retries": max(0, int(max_retries)),
            "created_at": time.time(),
            "history": [],
            "consecutive_failures": 0,
        }

        if schedule_type == "once":
            if not run_at:
                return ToolResult(success=False, error="A 'once' task needs run_at.")
            try:
                when = datetime.fromisoformat(run_at.replace("Z", ""))
            except ValueError:
                return ToolResult(success=False, error=(
                    f"Couldn't read run_at={run_at!r}. Use an ISO datetime like '2026-03-01T09:00'."
                ))
            if when.timestamp() <= time.time():
                return ToolResult(success=False, error=f"{run_at} is in the past.")
            task["run_at"] = when.timestamp()

        task["next_run"] = compute_next_run(task)
        if task["next_run"] is None:
            return ToolResult(success=False, error="That schedule never fires - check the settings.")

        tasks = load_tasks()
        tasks.append(task)
        save_tasks(tasks)
        return ToolResult(success=True, output={
            "task_id": task["id"],
            "name": task["name"],
            "schedule": describe_schedule(task),
            "next_run": datetime.fromtimestamp(task["next_run"]).strftime("%Y-%m-%d %H:%M"),
            "project": task["project"],
        })


class ListScheduledTasksTool(BaseTool):
    name = "list_scheduled_tasks"
    description = (
        "List scheduled tasks with their schedules, next run, last outcome and recent history. "
        "Use this to answer what's scheduled, what failed, and what a task produced."
    )
    parameters = [
        ToolParameter(name="include_disabled", type="boolean", required=False,
                      description="Include disabled tasks (default true - a task disabled after "
                                  "failures is exactly what the user needs to see)."),
        ToolParameter(name="task_id", type="string", required=False,
                      description="Only this task, with its full history."),
    ]

    async def run(self, include_disabled: bool = True, task_id: str = "", **kwargs) -> ToolResult:
        tasks = load_tasks()
        if task_id:
            task = next((t for t in tasks if t["id"] == task_id), None)
            if not task:
                return ToolResult(success=False, error=f"No scheduled task with id '{task_id}'.")
            return ToolResult(success=True, output={
                **_public(task), "full_history": task.get("history", []),
            })

        visible = [t for t in tasks if include_disabled or t.get("enabled", True)]
        visible.sort(key=lambda t: (not t.get("enabled", True), t.get("next_run") or float("inf")))
        return ToolResult(success=True, output={
            "tasks": [_public(t) for t in visible],
            "total": len(tasks),
            "failing": [t["name"] for t in tasks if t.get("consecutive_failures", 0) > 0],
            "disabled_after_failures": [t["name"] for t in tasks if t.get("disabled_reason")],
        })


class UpdateScheduledTaskTool(BaseTool):
    name = "update_scheduled_task"
    description = (
        "Enable, disable, or change a scheduled task - its instruction, time, or schedule. "
        "Re-enabling a task that was disabled after repeated failures also clears its failure "
        "count, so it gets a clean run rather than being disabled again immediately."
    )
    parameters = [
        ToolParameter(name="task_id", type="string", description="Task to change."),
        ToolParameter(name="enabled", type="boolean", required=False, description="Turn it on or off."),
        ToolParameter(name="instruction", type="string", required=False, description="New instruction."),
        ToolParameter(name="at", type="string", required=False, description="New time of day, HH:MM."),
        ToolParameter(name="weekdays", type="array", items_type="string", required=False,
                      description="New weekdays, for weekly tasks."),
        ToolParameter(name="every_minutes", type="number", required=False,
                      description="New interval, for interval tasks."),
    ]

    async def run(self, task_id: str, enabled: Optional[bool] = None, instruction: str = "",
                  at: str = "", weekdays: Optional[List[str]] = None,
                  every_minutes: Optional[float] = None, **kwargs) -> ToolResult:
        tasks = load_tasks()
        for task in tasks:
            if task["id"] != task_id:
                continue
            if instruction:
                task["instruction"] = instruction
            if at:
                task["at"] = at
            if weekdays:
                unknown = [d for d in weekdays if str(d).lower() not in WEEKDAYS]
                if unknown:
                    return ToolResult(success=False, error=f"Unknown weekday(s): {unknown}.")
                task["weekdays"] = [str(d).lower() for d in weekdays]
            if every_minutes is not None:
                task["every_minutes"] = float(every_minutes)
            if enabled is not None:
                task["enabled"] = bool(enabled)
                if enabled:
                    task["consecutive_failures"] = 0
                    task.pop("disabled_reason", None)

            task["next_run"] = compute_next_run(task) if task.get("enabled", True) else None
            save_tasks(tasks)
            return ToolResult(success=True, output=_public(task))
        return ToolResult(success=False, error=f"No scheduled task with id '{task_id}'.")


class DeleteScheduledTaskTool(BaseTool):
    name = "delete_scheduled_task"
    description = "Delete a scheduled task and its run history."
    parameters = [ToolParameter(name="task_id", type="string", description="Task to delete.")]

    async def run(self, task_id: str, **kwargs) -> ToolResult:
        tasks = load_tasks()
        remaining = [t for t in tasks if t["id"] != task_id]
        if len(remaining) == len(tasks):
            return ToolResult(success=False, error=f"No scheduled task with id '{task_id}'.")
        removed = next(t for t in tasks if t["id"] == task_id)
        save_tasks(remaining)
        return ToolResult(success=True, output=f"Deleted scheduled task '{removed['name']}'.")


class RunScheduledTaskNowTool(BaseTool):
    name = "run_scheduled_task_now"
    description = (
        "Run a scheduled task immediately, without waiting for its schedule and without "
        "changing when it next runs. Use this to check a task does what the user expects "
        "before leaving it to run on its own."
    )
    parameters = [ToolParameter(name="task_id", type="string", description="Task to run now.")]

    async def run(self, task_id: str, **kwargs) -> ToolResult:
        if _RUNNER is None:
            return ToolResult(success=False, error=(
                "The scheduler isn't running in this session, so a task can't be triggered. "
                "Scheduled tasks run in GUI and voice modes."
            ))
        # Deliberately not awaited: the task runs through the orchestrator, and
        # this call is itself inside an orchestrator turn - awaiting it would
        # deadlock on the turn lock.
        asyncio.create_task(_RUNNER.execute(task_id, manual=True))
        return ToolResult(success=True, output=(
            f"Started task {task_id} now. Its result will appear when it finishes; "
            f"list_scheduled_tasks shows the outcome."
        ))


class SystemSchedulingTool(BaseTool):
    name = "system_scheduling"
    description = (
        "Control whether scheduled tasks run when Leti is closed. Without this they only "
        "fire while Leti is open, so a Monday-morning task waits for someone to launch the "
        "app. Enabling it registers a periodic check with this machine's own scheduler - "
        "cron on Linux, launchd on macOS, Task Scheduler on Windows - which starts Leti, "
        "runs anything due, and exits.\n"
        "Actions: status (what's set up now), enable, disable. Tell the user what unattended "
        "runs are allowed to do: by default they can read, compute and write files, but not "
        "send email, post anything, or delete anything - those still need them present."
    )
    parameters = [
        ToolParameter(name="action", type="string", enum=["status", "enable", "disable"],
                      description="What to do."),
        ToolParameter(name="interval_minutes", type="number", required=False,
                      description="How often to check for due tasks when enabling (default 5)."),
    ]

    async def run(self, action: str = "status", interval_minutes: int = 5, **kwargs) -> ToolResult:
        from core import system_scheduler

        loop = asyncio.get_running_loop()
        try:
            if action == "enable":
                result = await loop.run_in_executor(
                    None, system_scheduler.install, int(interval_minutes))
                if not result.get("ok"):
                    return ToolResult(success=False, error=result.get("error", "Couldn't install."))
                allowed = ", ".join(
                    get_settings().get("scheduler", {}).get("unattended_allows",
                                                            ["read", "execute", "modify"]))
                return ToolResult(success=True, output={
                    **result,
                    "unattended_allows": allowed,
                    "note": (
                        f"Scheduled tasks now run every {result['interval_minutes']} minutes even "
                        f"when Leti is closed. Unattended runs may perform: {allowed}. Anything "
                        f"else - sending email, deleting - is refused with a reason recorded in "
                        f"the task's history, since nobody is there to confirm it."
                    ),
                })
            if action == "disable":
                result = await loop.run_in_executor(None, system_scheduler.uninstall)
                if not result.get("ok"):
                    return ToolResult(success=False, error=result.get("error", "Couldn't remove."))
                return ToolResult(success=True, output={
                    **result,
                    "note": ("Scheduled tasks now only run while Leti is open. Existing tasks and "
                             "their history are untouched."),
                })
            return ToolResult(success=True, output=await loop.run_in_executor(
                None, system_scheduler.status))
        except Exception as e:
            return ToolResult(success=False, error=str(e))
