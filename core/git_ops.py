"""Git, as something Leti can be careful with.

Every call here goes through tools/command_runner.py's run_command, which execs
without a shell and never blocks the event loop. Nothing in this file interpolates
into a command string, so a branch called `; rm -rf ~` is a branch name.

The idea the whole file is built around is that the user's work and Leti's work
have to stay distinguishable. A checkpoint therefore records two things before
anything is modified: the commit HEAD was at, and the set of files that were
ALREADY dirty. That second list is what makes a rollback safe - Leti may undo a
file it changed that was clean when it started, and it may never touch a file the
user had already been editing. There is no stash, no reset --hard, and no branch
juggling: those move the user's work around, and moving somebody's uncommitted
work is how an assistant loses an afternoon of it.

Reading is free. Writing - commit, push, branch - is a tool call like any other
and is authorised by SafetyGuard before it gets here. This module refuses a few
things outright regardless of permission (see FORBIDDEN), because a force push is
not a thing Leti should be able to be talked into.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.command_runner import run_command

logger = logging.getLogger("leti.git")

GIT_TIMEOUT = 60.0
PUSH_TIMEOUT = 180.0
MAX_DIFF_CHARS = 12_000
MAX_FILES_LISTED = 200

# Arguments that rewrite somebody else's history or throw work away. Refused here
# rather than left to the permission layer: there is no request for which Leti
# force-pushing is the right answer, and "the user approved it" is exactly how
# that goes wrong.
FORBIDDEN = ("--force", "-f", "--force-with-lease", "--hard", "--delete", "-D")


class GitError(Exception):
    """A git command that could not be run, or that failed for a stated reason."""


async def _git(root: Path, *args: str, timeout: float = GIT_TIMEOUT) -> str:
    """One git command in one repository. Raises GitError with git's own reason."""
    result = await run_command(["git", "-C", str(root), *args], timeout=timeout)
    if result.not_found:
        raise GitError("git is not installed on this machine, so Leti cannot use it here.")
    if not result.ok:
        raise GitError(result.failure_reason() or f"git {' '.join(args)} failed")
    return result.stdout


async def is_repository(root: Path) -> bool:
    result = await run_command(["git", "-C", str(root), "rev-parse", "--git-dir"],
                               timeout=GIT_TIMEOUT)
    return result.ok


async def status(root: Path) -> Dict[str, Any]:
    """Branch, upstream, and what is dirty - in porcelain, which is parseable."""
    raw = await _git(root, "status", "--porcelain=v1", "--branch", "--untracked-files=normal")
    branch, upstream, ahead, behind = None, None, 0, 0
    changed: List[Dict[str, str]] = []

    for line in raw.splitlines():
        if line.startswith("## "):
            head = line[3:]
            match = re.match(r"(?P<branch>[^.\s]+)(?:\.\.\.(?P<upstream>\S+))?", head)
            if match:
                branch = match.group("branch")
                upstream = match.group("upstream")
            ahead = int(m.group(1)) if (m := re.search(r"ahead (\d+)", head)) else 0
            behind = int(m.group(1)) if (m := re.search(r"behind (\d+)", head)) else 0
            continue
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:].strip().strip('"')
        # A rename reads "old -> new"; the new name is the one that exists now.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.append({"path": path, "code": code,
                        "staged": code[0] not in " ?", "untracked": code == "??"})

    head = (await _git(root, "rev-parse", "HEAD")).strip() if changed is not None else ""
    return {
        "root": str(root), "branch": branch, "upstream": upstream,
        "ahead": ahead, "behind": behind, "head": head,
        "changed": changed[:MAX_FILES_LISTED],
        "dirty": bool(changed),
        "dirty_paths": sorted({c["path"] for c in changed}),
    }


async def diff(root: Path, paths: Optional[List[str]] = None,
               staged: bool = False, stat_only: bool = False) -> Dict[str, Any]:
    """What changed, bounded. A diff that fills the context is not a review."""
    args = ["diff"]
    if staged:
        args.append("--cached")
    if stat_only:
        args.append("--stat")
    args.append("--")
    args.extend(paths or [])
    text = await _git(root, *[a for a in args if a != "--" or paths])
    truncated = len(text) > MAX_DIFF_CHARS
    return {"diff": text[:MAX_DIFF_CHARS], "truncated": truncated,
            "characters": len(text),
            "note": ("Shortened - ask for specific paths to see the rest."
                     if truncated else None)}


async def head_commit(root: Path) -> str:
    return (await _git(root, "rev-parse", "HEAD")).strip()


async def log(root: Path, limit: int = 10, paths: Optional[List[str]] = None) -> List[Dict[str, str]]:
    args = ["log", f"-{max(1, min(int(limit), 50))}", "--pretty=format:%H%x1f%an%x1f%ar%x1f%s"]
    if paths:
        args.append("--")
        args.extend(paths)
    raw = await _git(root, *args)
    out = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            out.append({"sha": parts[0][:10], "author": parts[1], "when": parts[2],
                        "subject": parts[3]})
    return out


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
#
# A checkpoint is a note, not a commit and not a stash. Taking it changes nothing
# in the repository: the user's index, working tree and branch are exactly where
# they were. What it records is enough to undo Leti's own edits later without
# touching anything else.

_CHECKPOINTS: Dict[str, Dict[str, Any]] = {}
MAX_CHECKPOINTS = 8


async def checkpoint(root: Path, label: str = "") -> Dict[str, Any]:
    """Record where the repository was before Leti touches it.

    Nothing is committed, stashed or moved. The two things kept are the commit
    HEAD is at, and which files were already dirty - because those files are the
    user's, and a rollback must leave them exactly as they are.
    """
    state = await status(root)
    mark = {
        "id": f"cp-{int(time.time())}",
        "root": str(root),
        "label": (label or "before Leti's changes")[:120],
        "at": time.time(),
        "head": state["head"],
        "branch": state["branch"],
        # The user's work, as it stood. Leti will not revert any of these.
        "already_dirty": list(state["dirty_paths"]),
        "note": ("Nothing was committed or stashed. This records where the "
                 "repository was so Leti's own edits can be undone later."),
    }
    _CHECKPOINTS[str(root)] = mark
    if len(_CHECKPOINTS) > MAX_CHECKPOINTS:
        oldest = min(_CHECKPOINTS, key=lambda k: _CHECKPOINTS[k]["at"])
        _CHECKPOINTS.pop(oldest, None)
    return mark


def last_checkpoint(root: Path) -> Optional[Dict[str, Any]]:
    return _CHECKPOINTS.get(str(root))


def clear_checkpoints() -> None:
    _CHECKPOINTS.clear()


async def changes_since_checkpoint(root: Path) -> Dict[str, Any]:
    """Which of the current changes are Leti's, and which were already there."""
    mark = last_checkpoint(root)
    state = await status(root)
    if mark is None:
        return {"checkpoint": None, "leti_changed": [], "user_changed": state["dirty_paths"],
                "note": "No checkpoint was taken, so nothing can be attributed to Leti."}
    already = set(mark["already_dirty"])
    now = set(state["dirty_paths"])
    return {
        "checkpoint": mark["id"],
        "taken_at": time.strftime("%H:%M", time.localtime(mark["at"])),
        "head_moved": state["head"] != mark["head"],
        "leti_changed": sorted(now - already),
        "user_changed": sorted(now & already),
        "note": ("Files in user_changed were already modified before Leti started. "
                 "They are not Leti's to revert."),
    }


async def rollback(root: Path, paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Undo Leti's own edits, and only those.

    Refuses outright to revert a file the user was already editing when the
    checkpoint was taken, whatever it is asked. A rollback that can eat somebody's
    uncommitted work is not a safety feature.
    """
    mark = last_checkpoint(root)
    if mark is None:
        raise GitError("There is no checkpoint for this repository, so there is nothing "
                       "to roll back to. Leti will not guess which changes were its own.")
    attribution = await changes_since_checkpoint(root)
    mine = set(attribution["leti_changed"])
    wanted = set(paths) if paths else set(mine)

    refused = sorted(wanted & set(attribution["user_changed"]))
    targets = sorted(wanted & mine)
    if not targets:
        return {"reverted": [], "refused": refused, "checkpoint": mark["id"],
                "note": ("Nothing to roll back: none of those files were changed by Leti "
                         "after the checkpoint." if not refused else
                         "Refused - those files were already being edited before Leti "
                         "started, so they are the user's work, not Leti's.")}

    if mark["head"] != (await head_commit(root)):
        raise GitError(
            "The repository has been committed to since the checkpoint, so reverting "
            "files would undo more than Leti's edits. Inspect the diff and decide by hand.")

    tracked, untracked = [], []
    state = await status(root)
    untracked_now = {c["path"] for c in state["changed"] if c["untracked"]}
    for path in targets:
        (untracked if path in untracked_now else tracked).append(path)

    if tracked:
        await _git(root, "checkout", "--", *tracked)
    removed = []
    for path in untracked:
        # A file Leti created that the repository has never seen. Deleting it is
        # part of undoing Leti's work; it cannot be somebody else's edit.
        target = root / path
        try:
            if target.is_file():
                target.unlink()
                removed.append(path)
        except OSError as e:
            logger.warning(f"Couldn't remove {path} during rollback: {e}")

    return {"reverted": tracked, "removed": removed, "refused": refused,
            "checkpoint": mark["id"],
            "note": ("Leti's edits were undone. Files the user was already editing were "
                     "left exactly as they were." if not refused else
                     f"Left alone (the user's own work): {', '.join(refused)}.")}


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def _check_arguments(args: List[str]) -> None:
    for argument in args:
        if argument in FORBIDDEN:
            raise GitError(
                f"'{argument}' rewrites or discards work that is not Leti's to discard. "
                "Leti does not run that, whoever asks - do it by hand if you mean it.")


async def create_branch(root: Path, name: str, from_ref: str = "") -> Dict[str, Any]:
    clean = str(name or "").strip()
    if not re.fullmatch(r"[\w./-]{1,100}", clean) or clean.startswith("-"):
        raise GitError(f"'{name}' is not a usable branch name.")
    args = ["checkout", "-b", clean] + ([from_ref] if from_ref else [])
    _check_arguments(args)
    await _git(root, *args)
    return {"branch": clean, "from": from_ref or "the current branch"}


async def commit(root: Path, message: str, paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Stage the named paths and commit them. No -a, so nothing sweeps in by accident."""
    text = str(message or "").strip()
    if not text:
        raise GitError("A commit needs a message that says what changed.")
    if not paths:
        raise GitError("Name the files to commit. Leti does not commit everything that "
                       "happens to be dirty - some of it may not be its work.")
    _check_arguments(list(paths))
    await _git(root, "add", "--", *paths)
    await _git(root, "commit", "-m", text)
    return {"committed": sorted(paths), "message": text.splitlines()[0],
            "sha": (await head_commit(root))[:10]}


async def remote_state(root: Path, branch: str = "") -> Dict[str, Any]:
    """What the remote looks like NOW, fetched rather than remembered.

    This is what stands between a push and somebody else's work: the local view of
    a remote branch can be hours old, and pushing on top of it is how a teammate's
    commit disappears.
    """
    try:
        await _git(root, "fetch", "--quiet", timeout=PUSH_TIMEOUT)
        fetched = True
    except GitError as e:
        fetched, reason = False, str(e)
    state = await status(root)
    branch = branch or state["branch"] or ""
    out = {"branch": branch, "upstream": state["upstream"], "fetched": fetched,
           "ahead": state["ahead"], "behind": state["behind"]}
    if not fetched:
        out["problem"] = f"Could not reach the remote: {reason}"
        return out
    if state["behind"]:
        out["diverged"] = True
        out["note"] = (f"The remote branch has {state['behind']} commit(s) this checkout "
                       "does not. Pushing now would need a merge or a rebase first - stop "
                       "and tell the user rather than forcing anything.")
    return out


async def push(root: Path, branch: str = "", remote: str = "origin",
               set_upstream: bool = False) -> Dict[str, Any]:
    """Push, after checking the remote has not moved. Never forces.

    The caller is expected to have been authorised by SafetyGuard already; this
    adds the check that authorisation cannot make safe, which is whether the
    remote still looks like what the decision was made against.
    """
    state = await remote_state(root, branch)
    if state.get("diverged"):
        raise GitError(state["note"])
    if not state.get("fetched"):
        raise GitError(state.get("problem", "The remote could not be reached."))

    branch = branch or state["branch"] or ""
    if not branch:
        raise GitError("Leti could not tell which branch to push.")
    args = ["push"] + (["--set-upstream"] if set_upstream or not state["upstream"] else [])
    args += [remote, branch]
    _check_arguments(args)
    await _git(root, *args, timeout=PUSH_TIMEOUT)
    return {"pushed": branch, "remote": remote,
            "upstream_set": set_upstream or not state["upstream"]}
