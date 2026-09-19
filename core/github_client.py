"""GitHub, over the API, with a token Leti can use but never say.

The credential lives in the Connections Manager the user already uses for their
mail, calendar and market keys (core/settings_editor.py, the "github" section),
marked secret - so it is written to config/settings.local.yaml at 0600 and is
echoed back to the interface as "(currently set)" and never as itself. Nothing in
this module returns it, logs it, puts it in a tool result or lets it reach the
model: it goes into an Authorization header and nowhere else, and _scrub() runs
over every error message on the way out in case GitHub ever echoes one back.

On authentication: this takes a fine-grained personal access token rather than
running an OAuth flow. That is not a shortcut, it is the narrower option. A
fine-grained token is scoped to the repositories the user picks and to the exact
permissions they tick - "contents: read" on one repository is a thing you can
make, and no OAuth scope is that small. An OAuth device flow would also need a
GitHub App registered under somebody's account and its client id shipped in the
source, which is a decision for the user rather than a default. No password is
ever asked for and none could be used.

Reading is ordinary. Writing - a branch, a file, a pull request - is a tool call
that SafetyGuard has already authorised before anything here runs, and this adds
the checks authorisation cannot make: whether the branch is protected, and
whether the remote still looks like what the decision was made against.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.github")

API = "https://api.github.com"
TIMEOUT = 30.0
MAX_FILE_BYTES = 400_000          # a file bigger than this is read in pieces
MAX_LISTED = 100
USER_AGENT = "Leti"


class GitHubError(Exception):
    """A GitHub request that failed, with why - and never with the token in it."""

    def __init__(self, message: str, kind: str = "error", status: int = 0):
        super().__init__(_scrub(message))
        self.kind = kind
        self.status = status


def _scrub(text: str) -> str:
    """Anything token-shaped, removed. Belt and braces: nothing should put one here."""
    text = re.sub(r"\b(gh[pousr]_[A-Za-z0-9]{16,})\b", "[token removed]", str(text))
    return re.sub(r"\b(github_pat_[A-Za-z0-9_]{20,})\b", "[token removed]", text)


def _settings() -> Dict[str, Any]:
    from core.config_loader import get_settings

    return get_settings().get("github", {}) or {}


def is_connected() -> bool:
    """Whether a token is configured. Never returns or logs the token itself."""
    return bool(str(_settings().get("token", "")).strip())


def default_repository() -> str:
    return str(_settings().get("repository", "") or "").strip()


def _headers() -> Dict[str, str]:
    token = str(_settings().get("token", "")).strip()
    if not token:
        raise GitHubError(
            "GitHub is not connected. Add a fine-grained personal access token under "
            "Connections (the 'github' section) - give it access to just the "
            "repositories you want Leti to see, with Contents: read (and write only if "
            "you want Leti to be able to push).", kind="unconfigured")
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": USER_AGENT}


async def _request(method: str, path: str, params: Optional[Dict[str, Any]] = None,
                   body: Optional[Dict[str, Any]] = None) -> Any:
    import httpx

    url = path if path.startswith("http") else f"{API}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.request(method, url, headers=_headers(),
                                            params=params or None, json=body)
    except GitHubError:
        raise
    except Exception as e:
        raise GitHubError(f"Couldn't reach GitHub: {e}", kind="network")

    if response.status_code in (200, 201):
        return response.json() if response.content else {}
    if response.status_code == 204:
        return {}

    detail = ""
    try:
        detail = str(response.json().get("message", ""))
    except Exception:
        detail = response.text[:200]

    if response.status_code == 401:
        raise GitHubError("GitHub rejected the token. It may have expired or been revoked - "
                          "replace it under Connections.", kind="auth", status=401)
    if response.status_code == 403:
        if "rate limit" in detail.lower():
            raise GitHubError("GitHub is rate-limiting this token; try again shortly.",
                              kind="rate_limit", status=403)
        raise GitHubError(
            f"The token is not permitted to do that ({detail}). A fine-grained token only "
            "reaches the repositories and permissions it was given.", kind="forbidden", status=403)
    if response.status_code == 404:
        raise GitHubError(
            f"Not found, or not visible to this token ({detail}). A fine-grained token "
            "sees only the repositories it was granted.", kind="not_found", status=404)
    if response.status_code == 409:
        raise GitHubError(f"GitHub says the state has moved on: {detail}", kind="conflict",
                          status=409)
    if response.status_code == 422:
        raise GitHubError(f"GitHub refused that as invalid: {detail}", kind="invalid", status=422)
    raise GitHubError(f"GitHub returned {response.status_code}: {detail}",
                      kind="error", status=response.status_code)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

async def account() -> Dict[str, Any]:
    """Who the token belongs to. Deliberately the only identity call there is."""
    me = await _request("GET", "/user")
    return {"login": me.get("login"), "name": me.get("name"),
            "account_type": me.get("type"), "connected": True}


async def repositories(limit: int = 30) -> List[Dict[str, Any]]:
    """The repositories this token can see - which is not "all of them"."""
    found = await _request("GET", "/user/repos",
                           {"per_page": max(1, min(int(limit), MAX_LISTED)),
                            "sort": "pushed"})
    return [_repo_summary(r) for r in (found or [])]


def _repo_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {"full_name": raw.get("full_name"), "private": raw.get("private"),
            "default_branch": raw.get("default_branch"),
            "description": (raw.get("description") or "")[:200],
            "language": raw.get("language"), "pushed_at": raw.get("pushed_at"),
            "permissions": {k: v for k, v in (raw.get("permissions") or {}).items()
                            if k in ("push", "pull", "admin")}}


async def repository(full_name: str) -> Dict[str, Any]:
    return _repo_summary(await _request("GET", f"/repos/{_repo(full_name)}"))


def _repo(full_name: str) -> str:
    name = str(full_name or "").strip().strip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", name):
        raise GitHubError(f"'{full_name}' is not an owner/repository name.", kind="invalid")
    return name


async def branches(full_name: str, limit: int = 30) -> List[Dict[str, Any]]:
    found = await _request("GET", f"/repos/{_repo(full_name)}/branches",
                           {"per_page": max(1, min(int(limit), MAX_LISTED))})
    return [{"name": b.get("name"), "protected": bool(b.get("protected")),
             "sha": (b.get("commit") or {}).get("sha", "")[:10]} for b in (found or [])]


async def is_protected(full_name: str, branch: str) -> bool:
    """Whether GitHub says this branch is protected. Asked before writing to it."""
    try:
        info = await _request("GET", f"/repos/{_repo(full_name)}/branches/{branch}")
    except GitHubError as e:
        if e.kind == "not_found":
            return False
        raise
    return bool(info.get("protected"))


async def contents(full_name: str, path: str = "", ref: str = "") -> Dict[str, Any]:
    """A directory listing, or one file's metadata. Never the whole repository."""
    found = await _request("GET", f"/repos/{_repo(full_name)}/contents/{path.strip('/')}",
                           {"ref": ref} if ref else None)
    if isinstance(found, list):
        return {"kind": "directory", "path": path or "/",
                "entries": [{"name": e.get("name"), "path": e.get("path"),
                             "type": e.get("type"), "size": e.get("size")}
                            for e in found[:MAX_LISTED]],
                "truncated": len(found) > MAX_LISTED}
    return {"kind": "file", "path": found.get("path"), "size": found.get("size"),
            "sha": found.get("sha")}


async def read_file(full_name: str, path: str, ref: str = "",
                    max_chars: int = 20_000) -> Dict[str, Any]:
    """One file's text, bounded. Large files come back with what was read said plainly."""
    found = await _request("GET", f"/repos/{_repo(full_name)}/contents/{path.strip('/')}",
                           {"ref": ref} if ref else None)
    if isinstance(found, list):
        raise GitHubError(f"{path} is a directory, not a file.", kind="invalid")
    size = int(found.get("size") or 0)
    if size > MAX_FILE_BYTES:
        raise GitHubError(
            f"{path} is {size:,} bytes, which is too big to read in one call. Ask for a "
            "specific part of it, or read it locally.", kind="too_large")
    encoded = found.get("content") or ""
    try:
        text = base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception as e:
        raise GitHubError(f"Couldn't decode {path}: {e}", kind="invalid")
    return {"path": found.get("path"), "sha": found.get("sha"), "bytes": size,
            "text": text[:max_chars], "truncated": len(text) > max_chars,
            "ref": ref or "the default branch"}


async def commits(full_name: str, branch: str = "", limit: int = 10,
                  path: str = "") -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"per_page": max(1, min(int(limit), 50))}
    if branch:
        params["sha"] = branch
    if path:
        params["path"] = path
    found = await _request("GET", f"/repos/{_repo(full_name)}/commits", params)
    return [{"sha": c.get("sha", "")[:10],
             "author": ((c.get("commit") or {}).get("author") or {}).get("name"),
             "when": ((c.get("commit") or {}).get("author") or {}).get("date"),
             "message": ((c.get("commit") or {}).get("message") or "").splitlines()[0][:120]}
            for c in (found or [])]


async def pull_requests(full_name: str, state: str = "open",
                        limit: int = 20) -> List[Dict[str, Any]]:
    found = await _request("GET", f"/repos/{_repo(full_name)}/pulls",
                           {"state": state, "per_page": max(1, min(int(limit), 50))})
    return [{"number": p.get("number"), "title": p.get("title"), "state": p.get("state"),
             "author": (p.get("user") or {}).get("login"),
             "head": (p.get("head") or {}).get("ref"),
             "base": (p.get("base") or {}).get("ref"),
             "draft": p.get("draft"), "url": p.get("html_url")} for p in (found or [])]


async def compare(full_name: str, base: str, head: str) -> Dict[str, Any]:
    found = await _request("GET", f"/repos/{_repo(full_name)}/compare/{base}...{head}")
    return {"status": found.get("status"), "ahead_by": found.get("ahead_by"),
            "behind_by": found.get("behind_by"),
            "files": [{"filename": f.get("filename"), "status": f.get("status"),
                       "additions": f.get("additions"), "deletions": f.get("deletions")}
                      for f in (found.get("files") or [])[:MAX_LISTED]]}


async def branch_head(full_name: str, branch: str) -> str:
    info = await _request("GET", f"/repos/{_repo(full_name)}/branches/{branch}")
    return ((info.get("commit") or {}).get("sha") or "")


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

async def create_branch(full_name: str, name: str, from_branch: str = "") -> Dict[str, Any]:
    repo = _repo(full_name)
    if not re.fullmatch(r"[\w./-]{1,100}", str(name or "")) or str(name).startswith("-"):
        raise GitHubError(f"'{name}' is not a usable branch name.", kind="invalid")
    base = from_branch or (await repository(full_name))["default_branch"]
    sha = await branch_head(full_name, base)
    await _request("POST", f"/repos/{repo}/git/refs",
                   body={"ref": f"refs/heads/{name}", "sha": sha})
    return {"branch": name, "from": base, "sha": sha[:10]}


async def create_pull_request(full_name: str, title: str, head: str, base: str = "",
                              body: str = "", draft: bool = False) -> Dict[str, Any]:
    """Open a pull request. Never merges one - that is somebody's decision, not a step."""
    repo = _repo(full_name)
    base = base or (await repository(full_name))["default_branch"]
    if head == base:
        raise GitHubError("A pull request needs two different branches.", kind="invalid")
    created = await _request("POST", f"/repos/{repo}/pulls", body={
        "title": str(title)[:250], "head": head, "base": base,
        "body": str(body)[:60_000], "draft": bool(draft)})
    return {"number": created.get("number"), "url": created.get("html_url"),
            "head": head, "base": base, "state": created.get("state")}
