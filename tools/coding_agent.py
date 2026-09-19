"""The tools that exist only in Coding Mode.

Four of them, deliberately. Each is one capability with an `action`, rather than
a tool per verb: fifteen small GitHub tools would be fifteen schemas in a context
window that has about three hundred tokens spare, and the model picks better from
four clear choices than from fifteen near-identical ones.

    coding_mode     enter it, leave it, say where it is working
    code_map        which files, which symbols, what refers to what, which tests
    git_workspace   status, checkpoint, diff, attribute, roll back, commit, push
    github          the repository side: read, inspect, branch, pull request

None of these is registered for Default Mode (see core/modes.py). A general
assistant turn cannot see them, is not offered them, and does not pay for their
schemas - which is the point of having modes at all.

Every one of them runs through the ordinary path: the router selects it, the
guard authorises it, the orchestrator executes it. Writing to a repository or
pushing to a remote is an `external` action in config/permissions.yaml and asks
before it happens, exactly like sending an email.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import coding, git_ops, github_client, modes
from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.coding_agent")


def _root(path: str = "", project: str = "") -> Path:
    """Where to work: what was asked for, the open workspace, or the active project."""
    if path:
        return resolve_path(path)
    space = coding.workspace()
    if space and space.root:
        return Path(space.root)
    from tools.coding import working_directory

    return working_directory(project, "")


class CodingModeTool(BaseTool):
    name = "coding_mode"
    description = (
        "Turn Coding Mode on or off, or report which mode Leti is in. Coding Mode is a "
        "software workspace: repository and symbol search, git checkpoints, targeted "
        "tests, GitHub. Only for 'enter/exit coding mode' or 'what mode are you in' - "
        "never switch just because a request mentions code. Keeps memory, projects, "
        "tasks and permissions as they are."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="enter, exit, or status.",
                      enum=["enter", "exit", "status"]),
        ToolParameter(name="project", type="string", required=False,
                      description="Project to work in when entering."),
        ToolParameter(name="path", type="string", required=False,
                      description="Folder to work in when entering, if not a project."),
        ToolParameter(name="repository", type="string", required=False,
                      description="GitHub repository as owner/name, if one is in play."),
    ]

    async def run(self, action: str, project: str = "", path: str = "",
                  repository: str = "", **kwargs) -> ToolResult:
        action = str(action or "status").lower()
        if action == "status":
            return ToolResult(success=True, output=modes.describe())

        if action == "exit":
            result = modes.leave()
            return ToolResult(success=True, output={
                **result, "note": "Back to the general assistant. The coding workspace is "
                                  "released; projects, memory and tasks are untouched."})

        if action != "enter":
            return ToolResult(success=False, error=f"'{action}' is not enter, exit or status.")

        result = modes.enter(modes.CODING)
        if not result.get("ok"):
            return ToolResult(success=False, error=result.get("error", "Couldn't switch mode."))

        root = ""
        try:
            root = str(_root(path, project))
        except Exception as e:
            logger.debug(f"No working folder resolved on entering coding mode: {e}")
        space = coding.open_workspace(root=root, repository=repository)

        details: Dict[str, Any] = {"working_in": space.root, "repository": space.repository}
        if space.root and Path(space.root).is_dir():
            try:
                if await git_ops.is_repository(Path(space.root)):
                    state = await git_ops.status(Path(space.root))
                    details["git"] = {"branch": state["branch"], "dirty": state["dirty"],
                                      "uncommitted_files": len(state["dirty_paths"])}
                else:
                    details["git"] = "not a git repository"
            except git_ops.GitError as e:
                details["git"] = str(e)
        details["github"] = ("connected" if github_client.is_connected()
                             else "not connected - add a token under Connections to use it")
        return ToolResult(success=True, output={**result, **details})


class CodeMapTool(BaseTool):
    name = "code_map"
    description = (
        "Understand a codebase before reading it: which files a request is about, what is "
        "defined in them, where a symbol is used, and which tests cover a change. Use this "
        "FIRST for anything beyond a one-file question - it is what turns 'fix the auth "
        "bug' into the three files worth opening, instead of reading the repository. "
        "Actions: files (rank files against a request), symbols (classes, functions, "
        "imports, routes in one file), references (where a name is used), tests (which "
        "tests relate to the files you changed). Reads nothing into context that you did "
        "not ask for."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="files, symbols, references, or tests.",
                      enum=["files", "symbols", "references", "tests"]),
        ToolParameter(name="request", type="string", required=False,
                      description="For files: what the user asked for, in their words."),
        ToolParameter(name="path", type="string", required=False,
                      description="For symbols: the file. Otherwise the folder to search."),
        ToolParameter(name="name", type="string", required=False,
                      description="For references: the class, function or constant."),
        ToolParameter(name="changed", type="array", items_type="string", required=False,
                      description="For tests: the files that changed, relative to the root."),
        ToolParameter(name="project", type="string", required=False,
                      description="Project to look in. Defaults to the open workspace."),
    ]

    async def run(self, action: str, request: str = "", path: str = "", name: str = "",
                  changed: Optional[List[str]] = None, project: str = "",
                  **kwargs) -> ToolResult:
        action = str(action or "").lower()

        if action == "symbols":
            if not path:
                return ToolResult(success=False, error="Which file's symbols?")
            target = resolve_path(path)
            if not target.is_file():
                root = _root("", project)
                target = root / path
            if not target.is_file():
                return ToolResult(success=False, error=f"No such file: {path}")
            return ToolResult(success=True, output=coding.symbols_in(target))

        root = _root(path if action in ("files", "references", "tests") and
                     Path(resolve_path(path) if path else ".").is_dir() else "", project)
        if not root.is_dir():
            return ToolResult(success=False, error=f"Not a folder: {root}")

        if action == "files":
            if not str(request).strip():
                return ToolResult(success=False,
                                  error="What is the request? Ranking needs something to rank against.")
            found = coding.find_files(root, request)
            return ToolResult(success=True, output={
                "root": str(root), "request": request, "files": found,
                "note": ("Ranked by name first, then by contents. Open the top few with "
                         "code_map symbols or read_file - do not read them all."
                         if found else
                         "Nothing matched. Try the words that would appear in a filename."),
            })

        if action == "references":
            if not str(name).strip():
                return ToolResult(success=False, error="Which name should Leti look for?")
            found = coding.references_to(root, name)
            return ToolResult(success=True, output={
                "root": str(root), "name": name, "references": found,
                "count": len(found),
                "note": ("Whole-word matches, capped. A definition and a call look the same "
                         "here - open the file to tell them apart."
                         if found else f"Nothing refers to '{name}' under {root}."),
            })

        if action == "tests":
            return ToolResult(success=True, output={
                "root": str(root), **coding.tests_for(root, changed or [])})

        return ToolResult(success=False,
                          error=f"'{action}' is not files, symbols, references or tests.")


class GitWorkspaceTool(BaseTool):
    name = "git_workspace"
    description = (
        "Work with the project's git repository safely. Actions: status (branch, what is "
        "dirty, ahead/behind), checkpoint (record where things were BEFORE you change "
        "anything - take one before any substantial edit), changes (which files are yours "
        "and which the user was already editing), diff (review before committing), log, "
        "rollback (undo only Leti's own edits), branch, commit, push. Rollback never "
        "touches a file the user was already editing, and push refuses when the remote has "
        "moved. Leti will not force-push, reset --hard or delete a branch at all."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="status, checkpoint, changes, diff, log, rollback, branch, "
                                  "commit, or push.",
                      enum=["status", "checkpoint", "changes", "diff", "log", "rollback",
                            "branch", "commit", "push"]),
        ToolParameter(name="path", type="string", required=False,
                      description="The repository. Defaults to the open workspace."),
        ToolParameter(name="message", type="string", required=False,
                      description="For commit: what changed and why, in the project's style."),
        ToolParameter(name="files", type="array", items_type="string", required=False,
                      description="For commit and rollback: which files. Required for commit."),
        ToolParameter(name="branch", type="string", required=False,
                      description="For branch and push: the branch name."),
        ToolParameter(name="project", type="string", required=False,
                      description="Project whose folder to use."),
    ]

    def action_case(self, arguments: Dict[str, Any]) -> str:
        """Reading a repository and writing to one are different acts.

        SafetyGuard classifies this per call (config/permissions.yaml,
        action_by_case), so `status` does not ask and `push` does.
        """
        action = str((arguments or {}).get("action", "")).lower()
        if action in ("commit", "push", "branch", "rollback"):
            return "write"
        return "read"

    async def run(self, action: str, path: str = "", message: str = "",
                  files: Optional[List[str]] = None, branch: str = "", project: str = "",
                  **kwargs) -> ToolResult:
        action = str(action or "").lower()
        root = _root(path, project)
        if not root.is_dir():
            return ToolResult(success=False, error=f"Not a folder: {root}")
        try:
            if not await git_ops.is_repository(root):
                return ToolResult(success=False, error=(
                    f"{root} is not a git repository, so there is nothing to checkpoint or "
                    "roll back to. Say so before changing anything substantial."))

            if action == "status":
                return ToolResult(success=True, output=await git_ops.status(root))
            if action == "checkpoint":
                mark = await git_ops.checkpoint(root, message)
                return ToolResult(success=True, output={
                    **mark, "next": "Make the changes. git_workspace changes will then say "
                                    "which files are Leti's."})
            if action == "changes":
                return ToolResult(success=True,
                                  output=await git_ops.changes_since_checkpoint(root))
            if action == "diff":
                return ToolResult(success=True, output=await git_ops.diff(root, files))
            if action == "log":
                return ToolResult(success=True, output={"commits": await git_ops.log(root)})
            if action == "rollback":
                return ToolResult(success=True, output=await git_ops.rollback(root, files))
            if action == "branch":
                return ToolResult(success=True, output=await git_ops.create_branch(root, branch))
            if action == "commit":
                return ToolResult(success=True,
                                  output=await git_ops.commit(root, message, files))
            if action == "push":
                return ToolResult(success=True, output=await git_ops.push(root, branch))
        except git_ops.GitError as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(success=False, error=f"'{action}' is not a git_workspace action.")


class GitHubTool(BaseTool):
    name = "github"
    description = (
        "Work with a GitHub repository the user's token can reach. Actions: account (who "
        "is connected), repos (what this token can see), repo, branches, files (browse a "
        "directory), read (one file's contents), commits, pulls, compare, create_branch, "
        "create_pull_request. Reading is how to review a repository without cloning it - "
        "browse, then read only the files that matter. Leti never merges a pull request, "
        "never force-pushes, and refuses to write to a protected branch. If GitHub is not "
        "connected, say so and point at Connections rather than guessing."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="account, repos, repo, branches, files, read, commits, "
                                  "pulls, compare, create_branch, create_pull_request.",
                      enum=["account", "repos", "repo", "branches", "files", "read",
                            "commits", "pulls", "compare", "create_branch",
                            "create_pull_request"]),
        ToolParameter(name="repository", type="string", required=False,
                      description="owner/name. Defaults to the one in settings."),
        ToolParameter(name="path", type="string", required=False,
                      description="For files and read: the path inside the repository."),
        ToolParameter(name="branch", type="string", required=False,
                      description="Branch to read from, or to create."),
        ToolParameter(name="base", type="string", required=False,
                      description="For compare and create_pull_request: the branch to "
                                  "compare or merge against."),
        ToolParameter(name="title", type="string", required=False,
                      description="For create_pull_request: the title."),
        ToolParameter(name="body", type="string", required=False,
                      description="For create_pull_request: summary, what was done, which "
                                  "tests ran and what they said, and known limitations."),
    ]

    def action_case(self, arguments: Dict[str, Any]) -> str:
        action = str((arguments or {}).get("action", "")).lower()
        return "write" if action in ("create_branch", "create_pull_request") else "read"

    async def run(self, action: str, repository: str = "", path: str = "", branch: str = "",
                  base: str = "", title: str = "", body: str = "", **kwargs) -> ToolResult:
        action = str(action or "").lower()
        space = coding.workspace()
        repo = (repository or (space.repository if space else "")
                or github_client.default_repository())

        if not github_client.is_connected():
            return ToolResult(success=False, error=(
                "GitHub is not connected. Add a fine-grained personal access token under "
                "Connections (the 'github' section), scoped to the repositories you want "
                "Leti to see - Contents: read is enough to review, and write only if you "
                "want it to be able to open a pull request."))

        needs_repo = action not in ("account", "repos")
        if needs_repo and not repo:
            return ToolResult(success=False, error=(
                "Which repository? Give it as owner/name, or set one in the github "
                "settings section."))

        try:
            if action == "account":
                return ToolResult(success=True, output=await github_client.account())
            if action == "repos":
                found = await github_client.repositories()
                return ToolResult(success=True, output={
                    "repositories": found, "count": len(found),
                    "note": "Only what this token was granted - not every repository the "
                            "account owns."})
            if action == "repo":
                return ToolResult(success=True, output=await github_client.repository(repo))
            if action == "branches":
                found = await github_client.branches(repo)
                return ToolResult(success=True, output={"repository": repo, "branches": found})
            if action == "files":
                return ToolResult(success=True, output={
                    "repository": repo, **await github_client.contents(repo, path, branch)})
            if action == "read":
                if not path:
                    return ToolResult(success=False, error="Which file should Leti read?")
                return ToolResult(success=True, output={
                    "repository": repo, **await github_client.read_file(repo, path, branch)})
            if action == "commits":
                found = await github_client.commits(repo, branch)
                return ToolResult(success=True, output={"repository": repo, "commits": found})
            if action == "pulls":
                found = await github_client.pull_requests(repo)
                return ToolResult(success=True, output={"repository": repo, "pull_requests": found})
            if action == "compare":
                if not (base and branch):
                    return ToolResult(success=False, error="Compare needs base and branch.")
                return ToolResult(success=True,
                                  output=await github_client.compare(repo, base, branch))
            if action == "create_branch":
                if not branch:
                    return ToolResult(success=False, error="What should the branch be called?")
                return ToolResult(success=True,
                                  output=await github_client.create_branch(repo, branch, base))
            if action == "create_pull_request":
                if not (branch and title):
                    return ToolResult(success=False,
                                      error="A pull request needs a branch and a title.")
                if await github_client.is_protected(repo, base or ""):
                    logger.info("Opening a PR against a protected branch - that is the "
                                "point of a PR, and merging it is still the user's call.")
                created = await github_client.create_pull_request(repo, title, branch, base, body)
                return ToolResult(success=True, output={
                    **created,
                    "note": "Opened, not merged. Merging is a separate decision and Leti "
                            "does not do it."})
        except github_client.GitHubError as e:
            return ToolResult(success=False, error=str(e), output={"problem": e.kind})
        return ToolResult(success=False, error=f"'{action}' is not a github action.")
